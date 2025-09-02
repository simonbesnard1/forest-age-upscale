import numpy as np

class AgeFusion:
    def __init__(self, cr_fuser, config):
        """
        Parameters
        ----------
        cr_fuser : CRBayesAgeFuser-like object
            Provides .run_map() for CR bias correction.
        config : dict
            Holds settings like TSD, max_age, sigma_TSD.
        """
        self.cr_fuser = cr_fuser
        self.config = config

    def correct_ml_age(self, ml_age, biomass, cr_params, cr_errors, ml_std, biomass_std, TSD, tmax):
        """Bias correct ML ages using CR priors."""
        valid = np.isfinite(ml_age) & np.isfinite(biomass)
        corrected = np.full_like(ml_age, np.nan, dtype=float)

        if valid.any():
            t_map, _ = self.cr_fuser.run_map(
                hat_t=ml_age[valid],
                B_obs=biomass[valid],
                A=cr_params["A"][valid],
                b=cr_params["b"][valid],
                k=cr_params["k"][valid],
                sd_A=cr_errors["A"][valid],
                sd_b=cr_errors["b"][valid],
                sd_k=cr_errors["k"][valid],
                sigma_ml=ml_std[valid],
                sigma_B_meas=biomass_std[valid],
                TSD=TSD[valid],
                tmax=tmax[valid],
                sigma_TSD=self.config.get("sigma_TSD", 5.0)
            )
            corrected[valid] = t_map
        return corrected

    def fuse(self, ML_pred_age_end, ML_pred_age_start, LTSD, biomass, cr_params, cr_errors, ml_std, biomass_std, TSD, tmax):
        """Fuse LTSD and ML ages with CR correction."""
        # 1. Correct ML ages (end year)
        ML_pred_age_end_corr = self.correct_ml_age(
            ML_pred_age_end, biomass, cr_params, cr_errors,
            ml_std, biomass_std, TSD, tmax
        )

        # 2. Start year: back-project unless LTSD overrides
        years_span = int(self.config["end_year"]) - int(self.config["start_year"])
        ML_pred_age_start_corr = ML_pred_age_end_corr - years_span
        too_young = ML_pred_age_start_corr < 1
        ML_pred_age_start_corr[too_young] = ML_pred_age_start[too_young]

        # 3. Apply LTSD overrides
        fused_end = np.where(LTSD > 0, LTSD, ML_pred_age_end_corr)
        fused_start = np.where(LTSD > 0, LTSD - years_span, ML_pred_age_start_corr)

        return fused_start, fused_end
