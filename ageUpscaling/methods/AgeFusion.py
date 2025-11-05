import numpy as np
from ageUpscaling.methods.CRBayesAgeFuser import AgeBiasCorrector


class AgeFusion:
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Holds settings like:
              - start_year, end_year (ints or str convertible)
              - sigma_TSD (float), default used by AgeBiasCorrector if per-pixel not provided
              - m_fixed (float, defaults to 1/3 inside corrector if omitted)
        """
        self.config = dict(config) if config is not None else {}
        self.sigma_TSD_default = float(self.config.get("sigma_TSD", 5.0))
        self.m_fixed = float(self.config.get("m_fixed", 0.67))
        self.units_scale = float(self.config.get("units_scale", 0.47))  # <<

    def _build_corrector(self, cr_params, cr_errors):
        """Instantiate the numba-based corrector for this tile."""
        return AgeBiasCorrector(
            A=cr_params["A"].astype(np.float32).reshape(-1),
            b=cr_params["b"].astype(np.float32).reshape(-1),
            k=cr_params["k"].astype(np.float32).reshape(-1),
            sd_A=cr_errors["A"].astype(np.float32).reshape(-1),
            sd_b=cr_errors["b"].astype(np.float32).reshape(-1),
            sd_k=cr_errors["k"].astype(np.float32).reshape(-1),
            m_fixed=self.m_fixed,
            sigma_TSD=self.sigma_TSD_default,
        )

    def correct_ml_age(
        self,
        ml_age, biomass,
        cr_params, cr_errors,
        ml_std, biomass_std,
        TSD, tmax
    ):
        """
        Bias-correct ML ages using CR likelihood (Numba Newton MAP).
        Inputs must already be in consistent units (e.g., if A is carbon,
        pass biomass scaled to carbon).
        """
        # flatten
        ml_age      = np.asarray(ml_age).reshape(-1)
        ml_std      = np.asarray(ml_std).reshape(-1)
        TSD         = np.asarray(TSD).reshape(-1)
        tmax        = np.asarray(tmax).reshape(-1)

        # convert biomass -> carbon (mean & std)
        biomass       = (np.asarray(biomass).reshape(-1) * self.units_scale).astype(np.float32)
        biomass_std = (np.asarray(biomass_std).reshape(-1) * self.units_scale).astype(np.float32)

        A = np.asarray(cr_params["A"]).reshape(-1)
        b = np.asarray(cr_params["b"]).reshape(-1)
        k = np.asarray(cr_params["k"]).reshape(-1)
        sd_A = np.asarray(cr_errors["A"]).reshape(-1)
        sd_b = np.asarray(cr_errors["b"]).reshape(-1)
        sd_k = np.asarray(cr_errors["k"]).reshape(-1)
        TSD  = np.asarray(TSD).reshape(-1)
        tmax = np.asarray(tmax).reshape(-1)

        # validity mask
        valid = (
            np.isfinite(ml_age) & np.isfinite(biomass) &
            np.isfinite(ml_std) & np.isfinite(biomass_std) &
            np.isfinite(A) & np.isfinite(b) & np.isfinite(k) &
            np.isfinite(sd_A) & np.isfinite(sd_b) & np.isfinite(sd_k) &
            np.isfinite(TSD) & np.isfinite(tmax)
        )

        corrected = np.full_like(ml_age, np.nan, dtype=float)
        if not valid.any():
            return corrected

        # Build corrector on-the-fly for this tile
        corrector = AgeBiasCorrector(A=A[valid], b=b[valid], k=k[valid],
                                     sd_A=sd_A[valid], sd_b=sd_b[valid], sd_k=sd_k[valid],
                                     m_fixed=self.m_fixed)

        # Precompute static sigma_param² at t = hat_t
        sigma_param2 = corrector.precompute_sigma_param2(hat_t=ml_age[valid].astype(np.float32))

        # Run fast Newton MAP
        t_map = corrector.correct(
            hat_t=ml_age[valid].astype(np.float32),
            B_obs=biomass[valid].astype(np.float32),
            sigma_ml=ml_std[valid].astype(np.float32),
            sigma_B_meas=biomass_std[valid].astype(np.float32),
            sigma_param2=sigma_param2.astype(np.float32),
            TSD=TSD[valid].astype(np.float32),
            tmax=tmax[valid].astype(np.float32),
            max_iter=15, tol=1e-3
        )

        corrected[valid] = t_map
        return corrected

    def fuse(
        self,
        ML_pred_age_start, ML_pred_age_end, LTSD,
        biomass_start, biomass_end,
        cr_params, cr_errors,
        ml_std_end, ml_std_start,
        biomass_std_end, biomass_std_start,
        TSD, tmax
    ):
        """
        Fuse LTSD rule with ML ages and CR-based bias correction.

        Notes
        -----
        - We correct end and start years independently (each with its own
          biomass and sigma), then enforce temporal consistency with the
          back-projection rule.
        - Pixels with LTSD < 50 are forced to LTSD (your rule), otherwise use corrected ML.
        - Final hard clip enforces [1, tmax].
        """
        years_span = int(self.config["end_year"]) - int(self.config["start_year"])

        # Enforce lower bound before correction to avoid pathological priors
        ML_pred_age_end   = np.maximum(np.asarray(ML_pred_age_end),   np.asarray(TSD))
        ML_pred_age_start = np.maximum(np.asarray(ML_pred_age_start), np.asarray(TSD))

        # --- Correct ML ages at end year
        ML_pred_age_end_corr = self.correct_ml_age(
            ML_pred_age_end, biomass_end, cr_params, cr_errors,
            ml_std_end, biomass_std_end, TSD, tmax
        )
        # If LTSD encodes "forest stable since 2000" as 50 -> take LTSD, else corrected
        fused_end = np.where(np.asarray(LTSD) < 50, np.asarray(LTSD, dtype=float), ML_pred_age_end_corr)

        # --- Correct ML ages at start year
        ML_pred_age_start_corr = self.correct_ml_age(
            ML_pred_age_start, biomass_start, cr_params, cr_errors,
            ml_std_start, biomass_std_start, TSD, tmax
        )

        # --- Back-project start year from fused_end
        fused_start = fused_end - years_span

        # Where the back-projection goes < 1, fallback to the corrected start ML
        too_young = (np.asarray(ML_pred_age_start) < 1)
        fused_start[too_young] = ML_pred_age_start_corr[too_young]

        # --- Final hard clip: enforce [1, tmax] everywhere
        fused_end   = np.clip(fused_end,   1, np.asarray(tmax))
        fused_start = np.clip(fused_start, 1, np.asarray(tmax))
        
        old_growth = (np.asarray(ML_pred_age_start) == 300)
        fused_start[old_growth] = 300
        fused_end[old_growth] = 300
        
        return fused_start, fused_end

