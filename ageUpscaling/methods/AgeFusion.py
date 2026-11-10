import numpy as np
from ageUpscaling.methods.CRBayesAgeFuser import AgeBiasCorrector


def reproject_ltsd_to_year(ltsd_anchor, target_year, anchor_year=2020):
    """Back-calculate an LTSD layer for an earlier year from the anchor year's export.

    The underlying Landsat/GEE disturbance sources (TMF, MapBiomas, Hansen,
    etc.) each report only the *most recent* qualifying event relative to
    the year they were run with (`BASELINE` in the GEE script), not full
    disturbance history. So this back-calculation implicitly assumes at
    most one disturbance event occurred in [target_year, anchor_year] per
    pixel -- reasonable over a ~15 year window (e.g. 2005-2020), much less
    so over the full monitored record (e.g. 1985-2020), where re-running
    the GEE export with `BASELINE=target_year` would be the rigorous
    alternative (it can recover an earlier event masked by the anchor
    year's product, which only remembers the latest one).

    Given `ltsd_anchor` (values in [1, 49] for a detected disturbance,
    or 50 for the "no disturbance in the full monitored record"
    sentinel):
      - sentinel (50): stays 50 -- no disturbance through the anchor year
        implies none through any earlier target_year either.
      - disturbed (< 50): the implied calendar disturbance year is
        `anchor_year - ltsd_anchor + 1`. If that's on or before
        target_year, the pixel had already experienced it, so its LTSD at
        target_year is `ltsd_anchor - (anchor_year - target_year)`. If the
        disturbance happened *after* target_year, we know a disturbance
        exists somewhere between target_year and anchor_year but not
        exactly when -- returned as `0.0` (a value real LTSD never takes),
        a distinct signal from "no information at all" (NaN, e.g.
        non-forest). `AgeFusion.fuse_multi_year` treats `0.0` specially:
        it forces the independent per-year ML/CR estimate and vetoes
        back-projection from the anchor year outright, since
        back-projecting a value that already reflects being young
        post-disturbance backward *past* a known (if imprecisely dated)
        disturbance has no sound physical interpretation -- unlike NaN,
        where back-projection absent any better information is a
        reasonable default.

    Parameters
    ----------
    ltsd_anchor : array-like
        LTSD values for the anchor year (e.g. the existing 2020 product).
    target_year : int
        The earlier calendar year to reproject to.
    anchor_year : int, default 2020
        The calendar year `ltsd_anchor` is valid for.

    Returns
    -------
    np.ndarray
        LTSD reprojected to `target_year`, same shape as `ltsd_anchor`.
        `0.0` marks "known intervening disturbance, timing unresolved"
        (see above); NaN marks genuinely missing input.
    """
    ltsd_anchor = np.asarray(ltsd_anchor, dtype=float)
    offset = anchor_year - target_year
    if offset < 0:
        raise ValueError(f"target_year ({target_year}) must be <= anchor_year ({anchor_year})")

    out = np.full_like(ltsd_anchor, np.nan)

    sentinel_mask = ltsd_anchor == 50
    out[sentinel_mask] = 50.0

    disturbed_mask = np.isfinite(ltsd_anchor) & (ltsd_anchor >= 1) & (ltsd_anchor < 50)
    candidate = ltsd_anchor - offset
    already_happened = disturbed_mask & (candidate >= 1)
    out[already_happened] = candidate[already_happened]
    postdates_target = disturbed_mask & ~already_happened
    out[postdates_target] = 0.0

    return out


class AgeFusion:

    PROVENANCE_LTSD_DIRECT = 0     # direct, severity-adjusted Landsat disturbance evidence
    PROVENANCE_OLD_GROWTH = 1      # Landsat-corroborated old-growth sentinel
    PROVENANCE_ML_REGRESSOR = 2    # ML regressor + CR bias correction (no direct Landsat evidence)
    PROVENANCE_BACK_PROJECTED = 3  # start year derived by offsetting the (valid) end-year fused age
    PROVENANCE_DISTURBANCE_MIXED = 4  # uncertain reset: mixture of LTSD-reset and retained-stand hypotheses
    PROVENANCE_NO_DATA = -1        # pixel invalid/no data (fused age itself is NaN)

    def __init__(self, config):
        """
        config keys (optional):
          - start_year, end_year
          - m_fixed: float (default 0.67)
          - units_scale: float (biomass -> carbon), default 0.47
          - old_growth_value: int sentinel for old growth in ML_pred_age_start (default 300)
          - og_proba_threshold: float, classifier probability threshold used to
            flag old growth on Landsat-corroborated (LTSD==50) pixels when a
            probability is supplied to `fuse` (default 0.3). Deliberately
            lower than the classifier's own 0.5 decision boundary: the
            classifier is trained on biomass/canopy-height/climate features
            and under-flags genuinely old but naturally low-biomass stands
            (e.g. poor soils, permafrost) the same way the age regressor
            does. A 50-year disturbance-free Landsat record is strong
            independent evidence, so less is asked of the classifier for
            those pixels rather than requiring it to independently clear 0.5.
          - replacement_probability_min/max: lower/upper bounds for the
            probability that a detected Landsat event reset stand age.
          - replacement_z_midpoint/slope: logistic mapping from positive
            LTSD biomass-mismatch z-score to replacement probability.
          - replacement_direct_threshold: probability above which provenance
            remains LTSD_DIRECT; lower values are DISTURBANCE_MIXED.
          - ml_sigma_floor: minimum global-model age uncertainty in years.
        """
        self.config = dict(config) if config is not None else {}
        self.m_fixed = float(self.config.get("m_fixed", 0.67))
        self.units_scale = float(self.config.get("units_scale", 0.47))
        self.old_growth_value = int(self.config.get("old_growth_value", 300))
        self.og_proba_threshold = float(self.config.get("og_proba_threshold", 0.3))
        self.replacement_probability_min = float(self.config.get("replacement_probability_min", 0.05))
        self.replacement_probability_max = float(self.config.get("replacement_probability_max", 0.95))
        self.replacement_z_midpoint = float(self.config.get("replacement_z_midpoint", 1.0))
        self.replacement_z_slope = float(self.config.get("replacement_z_slope", 3.0))
        self.replacement_direct_threshold = float(self.config.get("replacement_direct_threshold", 0.8))
        self.ml_sigma_floor = float(self.config.get("ml_sigma_floor", 5.0))

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

    @staticmethod
    def _cr_sensitivities_scalar(age, A, b, k, m):
        """Scalar dB/dA, dB/db, dB/dk (no numba, mirrors
        CRBayesAgeFuser._cr_sensitivities_at_t0; used for delta-method
        biomass uncertainty at a fixed age in the fusion logic)."""
        eps = 1e-9
        p = 1.0 / (1.0 - m)
        u = 1.0 - b * np.exp(-k * age)
        u = np.clip(u, eps, 1.0 - eps)
        dA = u ** p
        db = -A * p * (u ** (p - 1.0)) * np.exp(-k * age)
        dk = A * p * (u ** (p - 1.0)) * (b * age * np.exp(-k * age))
        return dA, db, dk

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
        TSD, tmax,
        apply_ml_sigma_floor=True,
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

        # Ensemble agreement is not equivalent to zero predictive error and
        # a zero value is singular in the MAP objective.
        if apply_ml_sigma_floor:
            ml_std = np.maximum(ml_std, self.ml_sigma_floor)
        else:
            ml_std = np.maximum(ml_std, 1e-3)

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

    def stand_replacement_probability(self, severity_z, disturbance_mask):
        """Convert LTSD/biomass consistency into P(stand replacement).

        A positive severity z means that more biomass remains than expected
        for a stand established at the Landsat event date.  It is therefore
        evidence for a partial event, and replacement probability decreases
        monotonically with positive z.  Negative z is capped at zero because
        low biomass alone cannot make the event more certain than the
        configured maximum.
        """
        z = self._as1d(severity_z).astype(float)
        disturbance_mask = self._as1d(disturbance_mask).astype(bool)
        out = np.full(z.shape, np.nan, dtype=np.float32)

        finite = disturbance_mask & np.isfinite(z)
        z_positive = np.maximum(z[finite], 0.0)
        exponent = np.clip(
            self.replacement_z_slope * (z_positive - self.replacement_z_midpoint),
            -60.0, 60.0,
        )
        logistic = 1.0 / (1.0 + np.exp(exponent))
        span = self.replacement_probability_max - self.replacement_probability_min
        out[finite] = self.replacement_probability_min + span * logistic

        # With no biomass/CR evidence, retain the direct Landsat hypothesis.
        missing = disturbance_mask & ~np.isfinite(z)
        out[missing] = self.replacement_probability_max
        return out

    def fuse(
        self,
        ML_pred_age_start, ML_pred_age_end, LTSD,
        biomass_start, biomass_end,
        cr_params, cr_errors,
        ml_std_end, ml_std_start,
        biomass_std_end, biomass_std_start,
        TSD, tmax,
        ml_old_growth_proba_end=None,
        ml_extrapolation_index_end=None,
        ml_extrapolation_index_start=None,
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

        Age floor from the disturbance record:
          LTSD is itself direct, biomass-independent evidence of a minimum
          age: a detected disturbance LTSD years ago means the stand cannot
          be younger than that, and the LTSD==50 sentinel (no disturbance
          seen anywhere in the monitored record) means age >= 50. This is
          used as a hard lower bound (TSD) in the CR/biomass bias
          correction below, so that naturally low-biomass but
          long-undisturbed stands cannot be pulled down towards "young" by
          the biomass-driven correction. Without it, a low-productivity
          old stand and a genuinely young stand with the same biomass are
          indistinguishable to the CR curve.

        Provenance and extrapolation:
          Each pixel's age comes from one of three mechanisms (see the
          PROVENANCE_* class constants), and "how much is this estimate
          extrapolated/untrustworthy" means something different for each,
          so a single feature-space distance metric would be meaningless
          for, say, a pixel whose age came directly from a detected
          disturbance. Both outputs are reported per pixel alongside age:
            - LTSD-direct: extrapolation = how much the severity check
              disagrees with the raw Landsat date (|severity_z| normalised
              by the z_suspicious threshold -- 0 means biomass exactly
              matches the CR curve's prediction at that disturbance date;
              1 means at or beyond the mismatch that maximally widens the
              prior sigma fed into the CR bias correction, see
              adjust_ltsd_with_severity).
            - Old-growth sentinel: extrapolation = 1 - classifier
              probability (0 = fully confident; only meaningful if
              `ml_old_growth_proba_end` is supplied, else 0 as a neutral
              default since the hard-threshold fallback already implies
              proba >= 0.5).
            - ML regressor + CR correction: extrapolation = the precomputed
              feature-space dissimilarity index passed in via
              `ml_extrapolation_index_end`/`_start` (see
              ExtrapolationIndex), already scaled to [0, 1]. NaN if not
              supplied -- this is the one branch where feature-space
              distance is actually the right question.
            - Back-projected start: inherits the end year's provenance
              signal (the assumption doing the work is "no disturbance
              intervened", which is a validity check, not a distance
              question -- there is nothing else to measure).
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

        # Disturbance-record-based hard floors (biomass-independent).
        ltsd_floor = np.where(np.isfinite(LTSD), np.clip(LTSD, 1.0, tmax), 1.0)
        TSD_end   = np.maximum(TSD, ltsd_floor)
        TSD_start = np.maximum(TSD, ltsd_floor - years_span)
        TSD_start = np.maximum(TSD_start, 1.0)

        # ===============================================================
        # 1a. Severity-aware adjustment of LTSD for LTSD < 50
        # ===============================================================
        # Only meaningful where LTSD < 50 and biomass_end is valid
        LTSD_eff, severity_z, sigma_ltsd = self.adjust_ltsd_with_severity(
            LTSD=LTSD,
            biomass_end=self._as1d(biomass_end),
            biomass_std_end=self._as1d(biomass_std_end),
            cr_params=cr_params,
            cr_errors=cr_errors,
            tmax=tmax,
            m_fixed=self.m_fixed,
        )

        # Where LTSD < 50 and finite, use the (unshifted) LTSD as the prior mean;
        use_ltsd_mask = np.isfinite(LTSD) & (LTSD < 50)
        fused_end_init = np.where(use_ltsd_mask, LTSD_eff, ML_pred_age_end)

        # Prior uncertainty: LTSD-direct pixels get the severity-aware LTSD
        # sigma (widened, not point-shifted, when biomass disagrees with the
        # disturbance date -- see adjust_ltsd_with_severity); ML-regressor
        # pixels keep the ensemble spread as before.
        sigma_prior_end = np.where(use_ltsd_mask, sigma_ltsd, self._as1d(ml_std_end))

        # ===============================================================
        # 1b. Correct candidate end-year age (Landsat/ML + CR)
        # ===============================================================
        fused_end = self.correct_ml_age(
            fused_end_init, biomass_end, cr_params, cr_errors,
            sigma_prior_end, biomass_std_end, TSD_end, tmax
        )

        # Fallback: if correction failed (NaN) but ML prediction was valid, keep ML
        missing_corr = np.isnan(fused_end) & np.isfinite(ML_pred_age_end)
        if np.any(missing_corr):
            fused_end[missing_corr] = ML_pred_age_end[missing_corr]
        
        # NEW: If LTSD < 50 but ML is NaN ⇒ keep severity-adjusted LTSD
        missing_ml = use_ltsd_mask & np.isnan(ML_pred_age_end)
        if np.any(missing_ml):
            fused_end[missing_ml] = fused_end_init[missing_ml]        

        # ===============================================================
        # 2. Correct ML ages (start)
        # ===============================================================
        ML_pred_age_start_corr = self.correct_ml_age(
            ML_pred_age_start, biomass_start, cr_params, cr_errors,
            ml_std_start, biomass_std_start, TSD_start, tmax
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
        if ml_old_growth_proba_end is not None:
            proba = self._as1d(ml_old_growth_proba_end)
            og = (LTSD == 50) & np.isfinite(proba) & (proba >= self.og_proba_threshold)
        else:
            og = (LTSD == 50) & np.isfinite(ML_pred_age_end) & (
                ML_pred_age_end >= self.old_growth_value
            )

        fused_start[og] = float(self.old_growth_value)
        fused_end[og]   = float(self.old_growth_value)

        # ===============================================================
        # 4b. Provenance + extrapolation (see docstring for the rationale
        # behind using a different trust signal per mechanism).
        # ===============================================================
        z_suspicious = 2.0  # matches adjust_ltsd_with_severity's default above

        provenance_end = np.full(LTSD.shape, self.PROVENANCE_ML_REGRESSOR, dtype=np.int8)
        provenance_end[use_ltsd_mask] = self.PROVENANCE_LTSD_DIRECT
        provenance_end[og] = self.PROVENANCE_OLD_GROWTH

        extrapolation_end = np.full(LTSD.shape, np.nan, dtype=np.float32)
        severity_ei = np.abs(severity_z) / z_suspicious
        extrapolation_end[use_ltsd_mask] = np.clip(severity_ei[use_ltsd_mask], 0.0, 1.0)

        ml_branch_end = provenance_end == self.PROVENANCE_ML_REGRESSOR
        if ml_extrapolation_index_end is not None:
            ei_ml_end = self._as1d(ml_extrapolation_index_end)
            extrapolation_end[ml_branch_end] = np.clip(ei_ml_end[ml_branch_end], 0.0, 1.0)

        if ml_old_growth_proba_end is not None:
            extrapolation_end[og] = np.clip(1.0 - proba[og], 0.0, 1.0)
        else:
            extrapolation_end[og] = 0.0

        # Start year: back-projected pixels inherit the end year's signal
        # (the assumption doing the work -- "no disturbance intervened" --
        # is a validity check already handled by `too_young`, not a
        # distance question); the ML-fallback branch gets its own
        # precomputed index; old-growth mirrors the end year exactly.
        provenance_start = np.full(LTSD.shape, self.PROVENANCE_BACK_PROJECTED, dtype=np.int8)
        provenance_start[too_young] = self.PROVENANCE_ML_REGRESSOR
        provenance_start[og] = self.PROVENANCE_OLD_GROWTH

        extrapolation_start = np.full(LTSD.shape, np.nan, dtype=np.float32)
        extrapolation_start[~too_young] = extrapolation_end[~too_young]
        if ml_extrapolation_index_start is not None:
            ei_ml_start = self._as1d(ml_extrapolation_index_start)
            extrapolation_start[too_young] = np.clip(ei_ml_start[too_young], 0.0, 1.0)
        extrapolation_start[og] = extrapolation_end[og]

        # ===============================================================
        # 5. Shared mask and clipping
        # ===============================================================
        base_nan = ~np.isfinite(fused_end)
        fused_end   = np.clip(fused_end,   1, tmax)
        fused_start = np.clip(fused_start, 1, tmax)
        fused_end[base_nan]   = np.nan
        fused_start[base_nan] = np.nan
        provenance_end[base_nan] = self.PROVENANCE_NO_DATA
        provenance_start[base_nan] = self.PROVENANCE_NO_DATA
        extrapolation_end[base_nan] = np.nan
        extrapolation_start[base_nan] = np.nan

        return (
            fused_start, fused_end, severity_z,
            provenance_start, provenance_end,
            extrapolation_start, extrapolation_end,
        )

    def _fuse_year_core(
        self,
        ML_pred_age, LTSD_year,
        biomass, cr_params, cr_errors,
        ml_std, biomass_std,
        TSD, tmax,
        ml_old_growth_proba=None,
        ml_extrapolation_index=None,
    ):
        """
        Core single-year fusion: severity-aware LTSD blending + CR bias
        correction + disturbance-record floor + old-growth sentinel, using
        one calendar year's own evidence. Shared building block for
        `fuse_multi_year`. Kept separate from `fuse()` (the original
        two-year start/end API) so that code path is untouched.

        Also computes provenance/extrapolation for this year in isolation
        (see `fuse`'s docstring for the rationale); `fuse_multi_year`
        combines these across years (back-projection, old-growth override)
        the same way `fuse` combines start/end.

        Returns
        -------
        fused : np.ndarray
        use_ltsd_mask : np.ndarray (bool)
            True where this year had a directly detected disturbance
            (LTSD_year < 50) -- i.e. `fused` is direct evidence for this
            year, not merely an ML/CR fallback.
        TSD_eff : np.ndarray
            This year's own disturbance-record floor (for validating
            whether back-projecting into/out of this year is legitimate).
        og : np.ndarray (bool)
            True where the old-growth sentinel was applied for this year.
        provenance : np.ndarray (int8)
            PROVENANCE_* code for this year, in isolation (a caller doing
            back-projection may still override PROVENANCE_ML_REGRESSOR
            with PROVENANCE_BACK_PROJECTED where `use_ltsd_mask`/`og` are
            both False).
        extrapolation : np.ndarray (float32)
            0-1 trust signal matching `provenance` (see `fuse`).
        replacement_probability : np.ndarray (float32)
            Probability that a detected Landsat event was stand replacing;
            NaN where there is no directly dated event.
        """
        ML_pred_age = np.maximum(self._as1d(ML_pred_age), self._as1d(TSD))
        LTSD_year = self._as1d(LTSD_year)
        TSD = self._as1d(TSD)
        tmax = self._as1d(tmax)

        ltsd_floor = np.where(np.isfinite(LTSD_year), np.clip(LTSD_year, 1.0, tmax), 1.0)
        TSD_eff = np.maximum(TSD, ltsd_floor)

        z_suspicious = 2.0
        LTSD_eff, severity_z, _ = self.adjust_ltsd_with_severity(
            LTSD=LTSD_year, biomass_end=biomass, biomass_std_end=biomass_std,
            cr_params=cr_params, cr_errors=cr_errors, tmax=tmax,
            m_fixed=self.m_fixed, z_suspicious=z_suspicious,
        )
        # LTSD_year == 0 is the reproject_ltsd_to_year sentinel for "a
        # disturbance is known to exist but its timing relative to this
        # year is unresolved" -- excluded here (no direct age value to
        # use), and vetoes back-projection explicitly in fuse_multi_year.
        use_ltsd_mask = np.isfinite(LTSD_year) & (LTSD_year >= 1) & (LTSD_year < 50)
        # Evaluate two distinct event hypotheses rather than representing a
        # potentially bimodal question with one broad LTSD-centred prior:
        # stand replacement (age anchored at LTSD), and partial disturbance
        # (pre-existing stand age retained). Both receive the same local CR
        # calibration; the retained-biomass mismatch controls their weights.
        # The raster age is their probability-weighted expectation, while the
        # probability is returned separately so ambiguity is not lost and a
        # downstream ensemble may sample the two modes if desired.
        fused_retained = self.correct_ml_age(
            ML_pred_age, biomass, cr_params, cr_errors, self._as1d(ml_std),
            biomass_std, TSD_eff, tmax,
        )
        missing_retained = np.isnan(fused_retained) & np.isfinite(ML_pred_age)
        fused_retained[missing_retained] = ML_pred_age[missing_retained]

        ltsd_sigma_base = float(self.config.get("ltsd_sigma_base", 1.0))
        reset_sigma = np.full(LTSD_year.shape, ltsd_sigma_base, dtype=float)
        fused_reset = self.correct_ml_age(
            LTSD_eff, biomass, cr_params, cr_errors, reset_sigma,
            biomass_std, TSD_eff, tmax, apply_ml_sigma_floor=False,
        )
        missing_reset = np.isnan(fused_reset) & use_ltsd_mask
        fused_reset[missing_reset] = LTSD_eff[missing_reset]

        replacement_probability = self.stand_replacement_probability(
            severity_z, use_ltsd_mask
        )
        fused = fused_retained.copy()
        both = use_ltsd_mask & np.isfinite(fused_reset) & np.isfinite(fused_retained)
        fused[both] = (
            replacement_probability[both] * fused_reset[both]
            + (1.0 - replacement_probability[both]) * fused_retained[both]
        )
        reset_only = use_ltsd_mask & np.isfinite(fused_reset) & ~np.isfinite(fused_retained)
        fused[reset_only] = fused_reset[reset_only]

        if ml_old_growth_proba is not None:
            proba = self._as1d(ml_old_growth_proba)
            og = (LTSD_year == 50) & np.isfinite(proba) & (proba >= self.og_proba_threshold)
        else:
            og = (LTSD_year == 50) & np.isfinite(ML_pred_age) & (ML_pred_age >= self.old_growth_value)
        fused[og] = float(self.old_growth_value)

        base_nan = ~np.isfinite(fused)
        fused = np.clip(fused, 1, tmax)
        fused[base_nan] = np.nan

        provenance = np.full(LTSD_year.shape, self.PROVENANCE_ML_REGRESSOR, dtype=np.int8)
        provenance[use_ltsd_mask] = self.PROVENANCE_LTSD_DIRECT
        mixed = use_ltsd_mask & (replacement_probability < self.replacement_direct_threshold)
        provenance[mixed] = self.PROVENANCE_DISTURBANCE_MIXED
        provenance[og] = self.PROVENANCE_OLD_GROWTH

        extrapolation = np.full(LTSD_year.shape, np.nan, dtype=np.float32)
        extrapolation[use_ltsd_mask] = np.clip(
            1.0 - replacement_probability[use_ltsd_mask], 0.0, 1.0
        )

        ml_branch = provenance == self.PROVENANCE_ML_REGRESSOR
        if ml_extrapolation_index is not None:
            ei_ml = self._as1d(ml_extrapolation_index)
            extrapolation[ml_branch] = np.clip(ei_ml[ml_branch], 0.0, 1.0)

        if ml_old_growth_proba is not None:
            extrapolation[og] = np.clip(1.0 - proba[og], 0.0, 1.0)
        else:
            extrapolation[og] = 0.0

        provenance[base_nan] = self.PROVENANCE_NO_DATA
        extrapolation[base_nan] = np.nan

        return (fused, use_ltsd_mask, TSD_eff, og, provenance,
                extrapolation, replacement_probability)

    def fuse_multi_year(
        self,
        anchor_year,
        ML_pred_age_by_year, LTSD_by_year,
        biomass_by_year, biomass_std_by_year, ml_std_by_year,
        cr_params, cr_errors,
        TSD, tmax,
        ml_old_growth_proba_by_year=None,
        ml_extrapolation_index_by_year=None,
        return_replacement_probability=False,
    ):
        """
        Fuse forest age independently for an arbitrary set of calendar
        years (e.g. 2005, 2010, 2015, 2020), generalising `fuse()`'s
        two-year start/end design to N years, each backed by its own
        LTSD export (see dev GEE script: BASELINE re-run per year).

        Semantics per non-anchor year Y:
          - If year Y's own LTSD shows a detected disturbance
            (LTSD_Y < 50): use that directly (severity-aware, CR-corrected)
            -- the most direct evidence available for that year.
          - Otherwise (no disturbance recorded at Y), prefer
            back-projecting from the anchor year's fused age
            (anchor_fused - (anchor_year - Y)): a stand undisturbed at
            the anchor year was simply that many years younger at Y.
            This is only valid if nothing disturbed the stand between Y
            and the anchor; that case is self-detected by the
            back-projected value underflowing year Y's own
            disturbance-record floor (a stand-replacing event between Y
            and the anchor makes the anchor's fused age too small to
            have already existed at Y) -- the same underflow check
            `fuse()` already uses between its two years, generalised to
            arbitrary offsets.
          - If back-projection underflows, fall back to year Y's own
            independent ML/CR-corrected estimate (same role as
            `ML_pred_age_start_corr` in `fuse()`).
          - Year Y's own old-growth sentinel (from its own LTSD/
            classifier evidence) always takes precedence over a
            back-projected value, since it is a categorically different
            statement ("unquantifiably old") rather than a point
            estimate arithmetic should be allowed to perturb.

        Parameters
        ----------
        anchor_year : int
            The year treated as the primary, directly-fused estimate
            (e.g. 2020). All other years are derived relative to it.
        ML_pred_age_by_year, LTSD_by_year, biomass_by_year,
        biomass_std_by_year, ml_std_by_year : dict[int, array]
            Per-year 1D arrays, keyed by calendar year. Must include
            `anchor_year` and every other requested year (the set of
            requested years is taken from `LTSD_by_year`'s keys).
        cr_params, cr_errors : dict
            CR curve parameters/uncertainties (static across years).
        TSD, tmax : array
            External hard floor (e.g. 1) and ceiling (e.g. 301), same
            for every year -- the real per-year floor additionally
            incorporates that year's own LTSD (see `_fuse_year_core`).
        ml_old_growth_proba_by_year : dict[int, array], optional
            Per-year classifier probability for the relaxed old-growth
            threshold (see `fuse`). Falls back to the hard-coded
            `ML_pred_age >= old_growth_value` rule per year if omitted.
        ml_extrapolation_index_by_year : dict[int, array], optional
            Per-year precomputed [0, 1] feature-space dissimilarity index
            (see ExtrapolationIndex), used for the PROVENANCE_ML_REGRESSOR
            branch's extrapolation score. NaN where not supplied.
        return_replacement_probability : bool, default False
            Preserve the original three-dictionary return API unless True.
            When True, append a fourth per-year dictionary containing the
            stand-replacement probability diagnostic.

        Returns
        -------
        age, provenance, extrapolation[, replacement_probability] : dict[int, np.ndarray]
            Three or four dicts (one per output), each keyed by calendar year.
            `age` is clipped to [1, tmax]; `provenance` is a PROVENANCE_*
            code; `extrapolation` is the matching 0-1 trust score (see
            `fuse`'s docstring for what each mechanism's score means).
        """
        years = sorted(LTSD_by_year.keys())
        if anchor_year not in years:
            raise ValueError(f"anchor_year {anchor_year} must be one of the requested years {years}")

        TSD = self._as1d(TSD)
        tmax = self._as1d(tmax)

        proba_anchor = (
            ml_old_growth_proba_by_year.get(anchor_year) if ml_old_growth_proba_by_year is not None else None
        )
        ei_anchor = (
            ml_extrapolation_index_by_year.get(anchor_year) if ml_extrapolation_index_by_year is not None else None
        )
        (fused_anchor, _, _, _, provenance_anchor, extrapolation_anchor,
         replacement_probability_anchor) = self._fuse_year_core(
            ML_pred_age=ML_pred_age_by_year[anchor_year],
            LTSD_year=LTSD_by_year[anchor_year],
            biomass=biomass_by_year[anchor_year],
            cr_params=cr_params, cr_errors=cr_errors,
            ml_std=ml_std_by_year[anchor_year], biomass_std=biomass_std_by_year[anchor_year],
            TSD=TSD, tmax=tmax,
            ml_old_growth_proba=proba_anchor,
            ml_extrapolation_index=ei_anchor,
        )

        age_results = {anchor_year: fused_anchor}
        provenance_results = {anchor_year: provenance_anchor}
        extrapolation_results = {anchor_year: extrapolation_anchor}
        replacement_probability_results = {anchor_year: replacement_probability_anchor}

        for year in years:
            if year == anchor_year:
                continue

            offset = anchor_year - year
            proba_year = (
                ml_old_growth_proba_by_year.get(year) if ml_old_growth_proba_by_year is not None else None
            )
            ei_year = (
                ml_extrapolation_index_by_year.get(year) if ml_extrapolation_index_by_year is not None else None
            )

            (fused_independent, use_ltsd_mask_year, TSD_eff_year, og_year,
             provenance_independent, extrapolation_independent,
             replacement_probability_year) = self._fuse_year_core(
                ML_pred_age=ML_pred_age_by_year[year],
                LTSD_year=LTSD_by_year[year],
                biomass=biomass_by_year[year],
                cr_params=cr_params, cr_errors=cr_errors,
                ml_std=ml_std_by_year[year], biomass_std=biomass_std_by_year[year],
                TSD=TSD, tmax=tmax,
                ml_old_growth_proba=proba_year,
                ml_extrapolation_index=ei_year,
            )

            disturbance_pending = self._as1d(LTSD_by_year[year]) == 0

            back_projected = fused_anchor - offset
            back_projection_valid = (
                np.isfinite(back_projected) & (back_projected >= TSD_eff_year) & ~disturbance_pending
            )
            use_back_projection = ~use_ltsd_mask_year & back_projection_valid

            fused_year = np.where(
                use_ltsd_mask_year,
                fused_independent,
                np.where(back_projection_valid, back_projected, fused_independent),
            )
            provenance_year = np.where(
                use_back_projection, self.PROVENANCE_BACK_PROJECTED, provenance_independent
            ).astype(np.int8)
            extrapolation_year = np.where(
                use_back_projection, extrapolation_anchor, extrapolation_independent
            ).astype(np.float32)

            # Year Y's own old-growth determination always wins over a
            # back-projected arithmetic value (see docstring); already
            # baked into fused_independent/provenance_independent/
            # extrapolation_independent by _fuse_year_core, so re-apply it
            # here for the case where back-projection would otherwise have
            # been chosen instead.
            fused_year = np.where(og_year, float(self.old_growth_value), fused_year)
            provenance_year = np.where(og_year, self.PROVENANCE_OLD_GROWTH, provenance_year).astype(np.int8)
            extrapolation_year = np.where(og_year, extrapolation_independent, extrapolation_year).astype(np.float32)

            base_nan = ~np.isfinite(fused_year)
            fused_year = np.clip(fused_year, 1, tmax)
            fused_year[base_nan] = np.nan
            provenance_year[base_nan] = self.PROVENANCE_NO_DATA
            extrapolation_year[base_nan] = np.nan

            age_results[year] = fused_year
            provenance_results[year] = provenance_year
            extrapolation_results[year] = extrapolation_year
            replacement_probability_results[year] = replacement_probability_year

        if return_replacement_probability:
            return (age_results, provenance_results, extrapolation_results,
                    replacement_probability_results)
        return age_results, provenance_results, extrapolation_results

    def adjust_ltsd_with_severity(
        self,
        LTSD,
        biomass_end,
        biomass_std_end,
        cr_params,
        cr_errors,
        tmax,
        m_fixed=0.67,
        z_consistent=0.0,
        z_suspicious=2.0,
        ltsd_sigma_base=1.0,
        ltsd_sigma_severity_scale=40.0,
    ):
        """
        Severity-aware LTSD *uncertainty* adjustment.

        Rationale: a Landsat-detected disturbance date is direct, valuable
        evidence, but a partial (non-stand-replacing) disturbance --
        selective logging, wind-throw of a few trees, fire scorch without
        full mortality -- can trigger spectral change detection without
        resetting the stand's age/biomass trajectory to zero. When observed
        biomass is *higher* than the CR curve predicts for a stand that
        young, that is the signature of exactly this: the true 100m-footprint
        mean age is probably older than the disturbance date alone suggests
        -- never younger, since the disturbance record is still real
        evidence something happened at that date; this is deliberately a
        one-sided (asymmetric) adjustment.

        This used to express that by point-shifting LTSD itself toward the
        biomass-implied age. That shifted value was then fed into
        `correct_ml_age`, which *independently* pulls it toward the same
        biomass observation again via the CR likelihood -- double-counting
        one piece of evidence through two separate mechanisms. Instead, this
        returns LTSD *unchanged* as the prior mean and widens the prior's
        uncertainty when the mismatch is suspicious; `correct_ml_age` then
        does the (single) reconciliation with biomass.

        Severity is measured as a z-score, not a ratio: earlier this used
        `R = B_obs / B_CR(LTSD)`, which blows up whenever B_CR is small --
        exactly the case for a genuinely fresh disturbance (LTSD ~1-3y),
        where the CR curve predicts near-zero biomass. Any realistic residual
        biomass (slash, fast regrowth, AGB-product noise) then sent R to a
        huge value and maximally widened the prior for precisely the
        population where the Landsat date should be trusted most. The
        z-score `z = (B_obs - B_CR(LTSD)) / sigma_diff` doesn't have this
        failure mode: `sigma_diff` combines biomass measurement uncertainty
        with the CR curve's own parameter uncertainty at that age (via the
        same delta-method sensitivities used in CRBayesAgeFuser), so a small
        absolute mismatch at a young age -- where both uncertainty sources
        are also small in absolute terms -- registers as unremarkable rather
        than as an extreme ratio. It also removes the old ratio thresholds'
        asymmetric ramp start (R<=0.9, not R==1, was "fully consistent"):
        z=0 is the true, symmetric point of exact agreement.

        Parameters
        ----------
        LTSD : 1D array
            Landsat time-since-disturbance (years); LTSD < 50 means disturbance detected.
        biomass_end : 1D array
            Observed biomass at end year [Mg C/ha or similar units].
        biomass_std_end : 1D array
            Biomass measurement uncertainty, same units as biomass_end.
        cr_params : dict of 1D arrays
            {"A", "b", "k"} CR parameters per pixel (same length as LTSD).
        cr_errors : dict of 1D arrays
            {"A", "b", "k"} marginal standard errors of the CR parameters
            (no covariance available -- see CRBayesAgeFuser.AgeBiasCorrector
            docstring for the same caveat).
        tmax : 1D array
            Maximum allowed age per pixel. Kept in the signature for
            compatibility with existing callers; no longer used to clip
            LTSD_eff now that LTSD is not shifted.
        m_fixed : float
            CR shape parameter m (same as in AgeBiasCorrector).
        z_consistent : float
            z at/below which observed biomass is at or below what the CR
            curve predicts for that age -- fully consistent with (or more
            severe than) the disturbance date -> full trust (sigma = base).
        z_suspicious : float
            z at/above which the mismatch is large enough, relative to its
            own uncertainty, to suspect a partial disturbance -> sigma
            inflated to its maximum. Default 2.0 (~one-sided 97.5% normal
            threshold).
        ltsd_sigma_base : float
            Prior std (years) attached to the LTSD date when z <= z_consistent.
            Reflects genuine Landsat disturbance-year detection precision,
            not biomass disagreement.
        ltsd_sigma_severity_scale : float
            Maximum *additional* std (years) added on top of the base when
            z >= z_suspicious, i.e. sigma_ltsd ranges over
            [ltsd_sigma_base, ltsd_sigma_base + ltsd_sigma_severity_scale].

        Returns
        -------
        LTSD_eff : 1D array
            LTSD unchanged (kept in the return signature so existing
            callers don't need to branch; no longer point-shifted).
        z : 1D array
            Severity z-score (for diagnostics).
        sigma_ltsd : 1D array
            Per-pixel prior std (years) to feed `correct_ml_age` for
            LTSD-direct pixels; `ltsd_sigma_base` where z is unavailable
            (LTSD>=50 or invalid), since it is unused there anyway.
        """
        LTSD = np.asarray(LTSD, dtype=np.float32).reshape(-1)
        biomass_end     = (self._as1d(biomass_end)     * self.units_scale).astype(np.float32)
        biomass_std_end = (self._as1d(biomass_std_end) * self.units_scale).astype(np.float32)

        tmax = np.asarray(tmax, dtype=np.float32).reshape(-1)

        A    = self._as1d(cr_params["A"])
        b    = self._as1d(cr_params["b"])
        k    = self._as1d(cr_params["k"])
        sd_A = self._as1d(cr_errors["A"])
        sd_b = self._as1d(cr_errors["b"])
        sd_k = self._as1d(cr_errors["k"])

        N = LTSD.size

        B_cr = np.full(N, np.nan, dtype=np.float32)
        sigma_cr = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if np.isfinite(LTSD[i]) and LTSD[i] > 0:
                B_cr[i] = self._cr_forward_scalar(LTSD[i], A[i], b[i], k[i], m_fixed)
                dA, db, dk = self._cr_sensitivities_scalar(LTSD[i], A[i], b[i], k[i], m_fixed)
                sigma_cr[i] = np.sqrt((dA * sd_A[i])**2 + (db * sd_b[i])**2 + (dk * sd_k[i])**2)

        sigma_diff = np.sqrt(biomass_std_end**2 + sigma_cr**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (biomass_end - B_cr) / np.maximum(sigma_diff, 1e-3)
        z[~np.isfinite(z)] = np.nan

        # LTSD is no longer shifted -- see rationale above.
        LTSD_eff = LTSD.astype(np.float32).copy()

        # Only informative where we actually have a disturbance and valid z
        mask = (LTSD < 50) & np.isfinite(z)

        # z <= z_consistent -> strong/consistent event -> tight prior (base sigma)
        # z >= z_suspicious -> suspicious mismatch -> maximally widened prior
        # z_consistent < z < z_suspicious -> interpolate linearly
        z_clipped = np.clip(z, z_consistent, z_suspicious)
        frac = (z_clipped - z_consistent) / max(z_suspicious - z_consistent, 1e-6)

        sigma_ltsd = np.full(N, np.float32(ltsd_sigma_base), dtype=np.float32)
        sigma_ltsd[mask] = ltsd_sigma_base + ltsd_sigma_severity_scale * frac[mask]

        return LTSD_eff, z, sigma_ltsd
