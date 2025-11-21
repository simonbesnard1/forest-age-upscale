import numpy as np
from ageUpscaling.methods.CRBayesAgeFuser import AgeBiasCorrector


class AgeFusion:
    def __init__(self, config):
        """
        config keys (optional):
          - start_year, end_year
          - m_fixed: float (default 0.67)
          - units_scale: float (biomass -> carbon), default 0.47
          - old_growth_value: int sentinel for old growth in ML_pred_age_start (default 300)
        """
        self.config = dict(config) if config is not None else {}
        self.m_fixed = float(self.config.get("m_fixed", 0.67))
        self.units_scale = float(self.config.get("units_scale", 0.47))
        self.old_growth_value = int(self.config.get("old_growth_value", 300))

    @staticmethod
    def _as1d(a):
        return np.asarray(a).reshape(-1)
    
    @staticmethod
    def _cr_forward_scalar(age, A, b, k, m):
        """Scalar CR forward (no numba, just for fusion logic)."""
        eps = 1e-9
        p = 1.0 / (1.0 - m)
        u = 1.0 - b * np.exp(-k * age)
        u = np.clip(u, eps, 1.0 - eps)
        return A * (u ** p)

    def _build_corrector(self, cr_params, cr_errors):
        """Instantiate the numba-based corrector for this tile."""
        return AgeBiasCorrector(
            A=self._as1d(cr_params["A"]).astype(np.float32),
            b=self._as1d(cr_params["b"]).astype(np.float32),
            k=self._as1d(cr_params["k"]).astype(np.float32),
            sd_A=self._as1d(cr_errors["A"]).astype(np.float32),
            sd_b=self._as1d(cr_errors["b"]).astype(np.float32),
            sd_k=self._as1d(cr_errors["k"]).astype(np.float32),
            m_fixed=self.m_fixed
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
        All inputs must be consistent with CR 'A' units (biomass converted to carbon here).
        """
        ml_age  = self._as1d(ml_age)
        ml_std  = self._as1d(ml_std)
        TSD     = self._as1d(TSD)
        tmax    = self._as1d(tmax)

        # biomass -> carbon (mean & std)
        biomass     = (self._as1d(biomass)     * self.units_scale).astype(np.float32)
        biomass_std = (self._as1d(biomass_std) * self.units_scale).astype(np.float32)

        A    = self._as1d(cr_params["A"])
        b    = self._as1d(cr_params["b"])
        k    = self._as1d(cr_params["k"])
        sd_A = self._as1d(cr_errors["A"])
        sd_b = self._as1d(cr_errors["b"])
        sd_k = self._as1d(cr_errors["k"])

        valid = (
            np.isfinite(ml_age) & np.isfinite(ml_std) &
            np.isfinite(biomass) & np.isfinite(biomass_std) &
            np.isfinite(A) & np.isfinite(b) & np.isfinite(k) &
            np.isfinite(sd_A) & np.isfinite(sd_b) & np.isfinite(sd_k) &
            np.isfinite(TSD) & np.isfinite(tmax)
        )

        corrected = np.full_like(ml_age, np.nan, dtype=float)
        if not valid.any():
            return corrected

        corrector = self._build_corrector(
            {"A": A[valid], "b": b[valid], "k": k[valid]},
            {"A": sd_A[valid], "b": sd_b[valid], "k": sd_k[valid]},
        )

        # Precompute sigma_param² at t = hat_t
        sigma_param2 = corrector.precompute_sigma_param2(
            hat_t=ml_age[valid].astype(np.float32)
        )

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
        Fuse Landsat LTSD with ML ages and CR-based bias correction.

        Semantics:
          - LTSD < 50  => disturbance detected within Landsat era:
                          use severity-aware LTSD (adjusted if biomass is too high).
          - LTSD == 50 => no disturbance detected since 2000:
                          use corrected ML for end year.
          - Back-project start from fused_end; if it underflows (<1),
                          fallback to corrected start ML.
          - Old-growth sentinel applied before clipping; final clip to [1, tmax].
        """
        start_year = int(self.config["start_year"])
        end_year   = int(self.config["end_year"])
        years_span = end_year - start_year

        # --- Flatten inputs
        ML_pred_age_end   = np.maximum(self._as1d(ML_pred_age_end),   self._as1d(TSD))
        ML_pred_age_start = np.maximum(self._as1d(ML_pred_age_start), self._as1d(TSD))
        LTSD = self._as1d(LTSD)
        TSD  = self._as1d(TSD)
        tmax = self._as1d(tmax)

        # ===============================================================
        # 1a. Severity-aware adjustment of LTSD for LTSD < 50
        # ===============================================================
        # Only meaningful where LTSD < 50 and biomass_end is valid
        LTSD_eff, R_severity = self.adjust_ltsd_with_severity(
            LTSD=LTSD,
            biomass_end=self._as1d(biomass_end),
            cr_params=cr_params,
            tmax=tmax,
            m_fixed=self.m_fixed,
            severity_lo=0.9,
            severity_hi=1.5,
            max_age_add=40.0,
        )

        # Where LTSD < 50 and finite, we use severity-adjusted LTSD_eff;
        use_ltsd_mask = np.isfinite(LTSD) & (LTSD < 50)
        fused_end_init = np.where(use_ltsd_mask, LTSD_eff, ML_pred_age_end)

        # ===============================================================
        # 1b. Correct candidate end-year age (Landsat/ML + CR)
        # ===============================================================
        fused_end = self.correct_ml_age(
            fused_end_init, biomass_end, cr_params, cr_errors,
            ml_std_end, biomass_std_end, TSD, tmax
        )

        # Fallback: if correction failed (NaN) but ML prediction was valid, keep ML
        missing_corr = np.isnan(fused_end) & np.isfinite(ML_pred_age_end)
        if np.any(missing_corr):
            fused_end[missing_corr] = ML_pred_age_end[missing_corr]

        # ===============================================================
        # 2. Correct ML ages (start)
        # ===============================================================
        ML_pred_age_start_corr = self.correct_ml_age(
            ML_pred_age_start, biomass_start, cr_params, cr_errors,
            ml_std_start, biomass_std_start, TSD, tmax
        )

        missing_corr_start = np.isnan(ML_pred_age_start_corr) & np.isfinite(ML_pred_age_start)
        if np.any(missing_corr_start):
            ML_pred_age_start_corr[missing_corr_start] = ML_pred_age_start[missing_corr_start]

        # ===============================================================
        # 3. Back-project start from fused_end
        # ===============================================================
        fused_start = fused_end - years_span

        # Replace where back-projection underflows
        too_young = (fused_start < 1)
        fused_start[too_young] = ML_pred_age_start_corr[too_young]

        # 3b. Fix pathological case: start NaN while end is valid
        missing_start = np.isnan(fused_start) & np.isfinite(fused_end)
        if np.any(missing_start):
            fused_start[missing_start] = ML_pred_age_start_corr[missing_start]

        # ===============================================================
        # 4. Old-growth sentinel before clipping (so it survives)
        # ===============================================================
        og = (LTSD == 50) & np.isfinite(ML_pred_age_end) & (
            ML_pred_age_end >= self.old_growth_value
        )

        fused_start[og] = float(self.old_growth_value)
        fused_end[og]   = float(self.old_growth_value)

        # ===============================================================
        # 5. Shared mask and clipping
        # ===============================================================
        base_nan = ~np.isfinite(fused_end)
        fused_end   = np.clip(fused_end,   1, tmax)
        fused_start = np.clip(fused_start, 1, tmax)
        fused_end[base_nan]   = np.nan
        fused_start[base_nan] = np.nan

        return fused_start, fused_end, R_severity
    
    
    def adjust_ltsd_with_severity(
        self,
        LTSD,
        biomass_end,
        cr_params,
        tmax,
        m_fixed=0.67,
        severity_lo=0.9,
        severity_hi=1.5,
        max_age_add=40.0,
    ):
        """
        Severity-aware LTSD adjustment.
    
        Parameters
        ----------
        LTSD : 1D array
            Landsat time-since-disturbance (years); LTSD < 50 means disturbance detected.
        biomass_end : 1D array
            Observed biomass at end year [Mg C/ha or similar units].
        cr_params : tuple of 1D arrays
            (A, b, k) CR parameters per pixel (same length as LTSD).
        tmax : 1D array
            Maximum allowed age per pixel (e.g. biome-specific cap).
        m_fixed : float
            CR shape parameter m (same as in your AgeBiasCorrector).
        severity_lo : float
            Lower ratio threshold R = B_obs / B_CR(LTSD) below which
            we consider the event "stand-replacing or strong".
            R <= severity_lo → keep LTSD unchanged.
        severity_hi : float
            Upper ratio threshold above which we consider LTSD "very suspicious".
            R >= severity_hi → add max_age_add years.
        max_age_add : float
            Maximum age *increase* added to LTSD when biomass is too high.
    
        Returns
        -------
        LTSD_eff : 1D array
            Severity-adjusted LTSD, clipped to [1, tmax].
        R : 1D array
            Biomass ratio R = B_obs / B_CR(LTSD) (for diagnostics).
        """
        LTSD = np.asarray(LTSD, dtype=np.float32).reshape(-1)
        biomass_end = np.asarray(biomass_end, dtype=np.float32).reshape(-1)
        biomass_end     = (self._as1d(biomass_end)     * self.units_scale).astype(np.float32)

        tmax = np.asarray(tmax, dtype=np.float32).reshape(-1)
    
        A    = self._as1d(cr_params["A"])
        b    = self._as1d(cr_params["b"])
        k    = self._as1d(cr_params["k"])
    
        eps = 1e-6
        N = LTSD.size
    
        B_cr = np.empty(N, dtype=np.float32)
        for i in range(N):
            if np.isfinite(LTSD[i]) and LTSD[i] > 0:
                B_cr[i] = self._cr_forward_scalar(LTSD[i], A[i], b[i], k[i], m_fixed)
            else:
                B_cr[i] = np.nan
    
        R = biomass_end / (B_cr + eps)
        # Guard against negative or nonsense biomass
        R[~np.isfinite(R)] = np.nan
    
        # Start from original LTSD
        LTSD_eff = LTSD.astype(np.float32).copy()
    
        # Only adjust where we actually have a disturbance and valid ratio
        mask = (LTSD < 50) & np.isfinite(R)
    
        # For R <= severity_lo → strong event, keep LTSD as-is
        # For R >= severity_hi → very suspicious, add max_age_add
        # For severity_lo < R < severity_hi → interpolate linearly
        R_clipped = np.clip(R, severity_lo, severity_hi)
        frac = (R_clipped - severity_lo) / max(severity_hi - severity_lo, 1e-6)
        delta_t = max_age_add * frac
    
        # Apply only where LTSD < 50 and R finite
        LTSD_eff[mask] = LTSD[mask] + delta_t[mask]
    
        # Clip to [1, tmax]
        LTSD_eff = np.maximum(1.0, np.minimum(LTSD_eff, tmax))
    
        return LTSD_eff, R



