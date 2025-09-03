import numpy as np
from ageUpscaling.methods.CRBayesAgeFuser import CRBayesAgeFuser

class AgeFusion:
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Holds settings like TSD, max_age, sigma_TSD.
        """
        self.config = config

    def correct_ml_age(self, ml_age, biomass, cr_params, cr_errors, ml_std, biomass_std, TSD, tmax):
        """Bias correct ML ages using CR priors."""
        valid = np.isfinite(ml_age) & np.isfinite(biomass) & \
                np.isfinite(cr_params["A"]) & np.isfinite(cr_params["b"]) & np.isfinite(cr_params["k"]) & \
                np.isfinite(ml_std) & np.isfinite(biomass_std) & \
                np.isfinite(cr_errors["A"]) & np.isfinite(cr_errors["b"]) & np.isfinite(cr_errors["k"])
        
        corrected = np.full_like(ml_age, np.nan, dtype=float)

        if valid.any():
            t_map, _ = CRBayesAgeFuser.run_map(
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

    def fuse(self, ML_pred_age_start, ML_pred_age_end, LTSD,
             biomass_start, biomass_end, cr_params, cr_errors,
             ml_std_end, ml_std_start, biomass_std_end, biomass_std_start, TSD, tmax):
        """Fuse LTSD and ML ages with CR correction."""
        
        years_span = int(self.config["end_year"]) - int(self.config["start_year"])

        # --- Enforce TSD lower bound on ML predictions before correction
        ML_pred_age_end = np.maximum(ML_pred_age_end, TSD)
        ML_pred_age_start = np.maximum(ML_pred_age_start, TSD)
    
        # --- Correct ML ages at end year
        ML_pred_age_end_corr = self.correct_ml_age(
            ML_pred_age_end, biomass_end, cr_params, cr_errors,
            ml_std_end, biomass_std_end, TSD, tmax
        )
        fused_end = np.where(LTSD < 50, LTSD, ML_pred_age_end_corr)
        
        # --- Correct ML ages at start year
        ML_pred_age_start_corr = self.correct_ml_age(
            ML_pred_age_start, biomass_start, cr_params, cr_errors,
            ml_std_start, biomass_std_start, TSD, tmax
        )
    
        # --- Back-project start year
        fused_start = fused_end - years_span
        too_young = ML_pred_age_start < 1
        fused_start[too_young] = ML_pred_age_start_corr[too_young]
        
        # --- Final hard clip: enforce TSD and tmax
        fused_end = np.clip(fused_end, 1, tmax)
        fused_start = np.clip(fused_start, 1, tmax)
    
        return fused_start, fused_end
