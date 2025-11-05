import numpy as np
from numba import njit, prange


# ==========================================================
# 0. NUMERICAL HELPERS
# ==========================================================

@njit(cache=True, fastmath=True, parallel=False)
def _safe_clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

@njit(cache=True, fastmath=True, parallel=False)
def _softplus(x):
    # stable softplus approximation
    if x > 20.0:
        return x
    if x < -20.0:
        return np.exp(x)
    return np.log1p(np.exp(x))

@njit(cache=True, fastmath=True, parallel=False)
def _dsoftplus(x):
    # derivative of softplus = sigmoid
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return np.exp(x)
    e = np.exp(x)
    return e / (1.0 + e)


# ==========================================================
# 1. CHAPMAN–RICHARDS FORWARD + DERIVATIVES
#    Your parameterization: B = A * (1 - b * exp(-k t))^(1/(1-m))
# ==========================================================

@njit(cache=True, fastmath=True, parallel=False)
def _cr_forward(t, A, b, k, m):
    eps = 1e-12
    p = 1.0 / (1.0 - m)
    u = 1.0 - b * np.exp(-k * t)
    if u < eps:
        u = eps
    if u > 1.0 - eps:
        u = 1.0 - eps
    return A * (u ** p)

@njit(cache=True, fastmath=True, parallel=False)
def _cr_derivative(t, A, b, k, m):
    eps = 1e-12
    p = 1.0 / (1.0 - m)
    u = 1.0 - b * np.exp(-k * t)
    if u < eps:
        u = eps
    if u > 1.0 - eps:
        u = 1.0 - eps
    return A * p * (u ** (p - 1.0)) * (b * k * np.exp(-k * t))

@njit(cache=True, fastmath=True, parallel=False)
def _cr_sensitivities_at_t0(t0, A, b, k, m):
    # gradients wrt A, b, k at t0 (for delta-method)
    eps = 1e-12
    p = 1.0 / (1.0 - m)
    u = 1.0 - b * np.exp(-k * t0)
    if u < eps:
        u = eps
    if u > 1.0 - eps:
        u = 1.0 - eps
    dA = u ** p
    db = -A * p * (u ** (p - 1.0)) * np.exp(-k * t0)
    dk =  A * p * (u ** (p - 1.0)) * (b * t0 * np.exp(-k * t0))
    return dA, db, dk

@njit(cache=True, fastmath=True, parallel=False)
def _cr_inverse(B, A, b, k, m):
    """
    Invert CR for t given B (clamped). Returns age consistent with biomass.
    t = -(1/k) * ln( (1 - (B/A)^(1-m)) / b )
    """
    eps = 1e-12
    # guard
    A_ = A if A > eps else eps
    b_ = b
    if b_ < eps:
        b_ = eps
    if b_ > 1.0 - eps:
        b_ = 1.0 - eps
    k_ = k if k > eps else eps

    r = B / A_
    if r < eps:
        r = eps
    if r > 1.0 - eps:
        r = 1.0 - eps

    q = 1.0 - r ** (1.0 - m)  # in (0,1)
    z = q / b_
    if z < eps:
        z = eps
    if z > 1.0 - eps:
        z = 1.0 - eps
    return -np.log(z) / k_


# ==========================================================
# 2. STATIC SIGMA_PARAM² (delta method, precomputed)
# ==========================================================

@njit(cache=True, fastmath=True, parallel=True)
def _precompute_sigma_param2(A, b, k, sd_A, sd_b, sd_k, hat_t, m):
    N = A.size
    out = np.empty(N, dtype=np.float32)
    for i in prange(N):
        dA, db, dk = _cr_sensitivities_at_t0(hat_t[i], A[i], b[i], k[i], m)
        out[i] = (dA*dA)*(sd_A[i]**2) + (db*db)*(sd_b[i]**2) + (dk*dk)*(sd_k[i]**2)
    return out


# ==========================================================
# 3. NEWTON SOLVERS
#    3a) Gamma slack prior in t-space (fast, robust)
#    3b) Softplus reparam t = TSD + softplus(s) (no boundary)
# ==========================================================

@njit(cache=True, fastmath=True, parallel=True)
def _newton_solver_slack_prior(
    hat_t, B_obs, sigma_ml, sigma_B_meas,
    A, b, k, sigma_param2, TSD, tmax,
    m, max_iter, tol, damping,
    k_shape, theta_slack,                  # k_shape -> alpha (barrier strength); theta_slack unused
    robust_delta_frac, sigma_eff_floor_frac,
    use_adaptive_variance, sd_A, sd_b, sd_k
):
    """
    MAP Newton solver with:
      - neutral lower bound via log-barrier  -alpha*log(t - TSD)
      - pseudo-Huber weighting on biomass residual
      - inverse-CR + ML blend initialization
      - optional adaptive variance where needed

    NOTE: k_shape is repurposed as the barrier strength alpha (typical 0.5–2.0).
          theta_slack is ignored (kept for API compatibility).
    """
    N = hat_t.size
    t = hat_t.copy().astype(np.float32)
    alpha = k_shape  # barrier strength

    for i in prange(N):
        # Clamp biomass to [0, A]
        B_target = B_obs[i]
        if B_target < 0.0:
            B_target = 0.0
        if B_target > A[i]:
            B_target = A[i]

        # Init from inverse-CR & ML blend (no directional slack)
        t_cr = _cr_inverse(B_target, A[i], b[i], k[i], m)
        ti = 0.5 * (hat_t[i] + t_cr)
        if ti < TSD[i] + 1e-3:
            ti = TSD[i] + 1e-3
        if ti > tmax[i]:
            ti = tmax[i]

        # Decide adaptive-variance
        do_adapt = use_adaptive_variance or (abs(t_cr - hat_t[i]) > 10.0)

        for _ in range(max_iter):
            # Variance (optionally adaptive)
            if do_adapt:
                dA, dbs, dk_ = _cr_sensitivities_at_t0(ti, A[i], b[i], k[i], m)
                sigma_param2_i = (dA*dA)*(sd_A[i]**2) + (dbs*dbs)*(sd_b[i]**2) + (dk_*dk_)*(sd_k[i]**2)
            else:
                sigma_param2_i = sigma_param2[i]

            # Forward model
            Bi   = _cr_forward(ti, A[i], b[i], k[i], m)
            diff = Bi - B_target
            dBdt = _cr_derivative(ti, A[i], b[i], k[i], m)

            # Effective variance with floor
            sigma_eff2 = sigma_B_meas[i]**2 + sigma_param2_i
            floor2 = (sigma_eff_floor_frac * A[i])**2
            if sigma_eff2 < floor2:
                sigma_eff2 = floor2

            # Pseudo-Huber weights
            delta = robust_delta_frac * A[i]
            if delta < 1.0:
                delta = 1.0
            r = diff / np.sqrt(sigma_eff2)
            w_grad = 1.0 / np.sqrt(1.0 + (r / delta)**2)
            w_hess = w_grad  # conservative GN

            # Neutral lower-bound via log-barrier
            s = ti - TSD[i]
            if s < 1e-6:
                s = 1e-6
            g_barrier = -alpha / s
            h_barrier =  alpha / (s * s)

            # Gradient/Hessian of negative log-posterior wrt t
            g = (ti - hat_t[i]) / (sigma_ml[i]**2) \
                + (diff * dBdt) / sigma_eff2 * w_grad \
                + g_barrier

            h = 1.0 / (sigma_ml[i]**2) \
                + (dBdt * dBdt) / sigma_eff2 * w_hess \
                + h_barrier

            if h < 1e-12:
                break

            step  = damping * g / h
            ti_new = ti - step

            # Hard bounds
            if ti_new < TSD[i] + 1e-6:
                ti_new = TSD[i] + 1e-6
            if ti_new > tmax[i]:
                ti_new = tmax[i]

            if abs(step) < tol:
                ti = ti_new
                break
            ti = ti_new

        t[i] = ti

    return t


@njit(cache=True, fastmath=True, parallel=True)
def _newton_solver_softplus_reparam(
    hat_t, B_obs, sigma_ml, sigma_B_meas,
    A, b, k, sigma_param2, TSD, tmax,
    m, max_iter, tol, damping,
    mu_s, sd_s,                         # kept for API; not used (neutral)
    robust_delta_frac, sigma_eff_floor_frac,
    use_adaptive_variance, sd_A, sd_b, sd_k
):
    """
    MAP Newton solver in s-space with t = TSD + softplus(s).
    Neutral: no prior on s (softplus enforces the bound without bias).
    """
    N = hat_t.size
    t_out = np.empty(N, dtype=np.float32)

    for i in prange(N):
        # Clamp biomass
        B_target = B_obs[i]
        if B_target < 0.0:
            B_target = 0.0
        if B_target > A[i]:
            B_target = A[i]

        # Initialize from inverse-CR & ML blend (tiny offset above TSD)
        t_cr = _cr_inverse(B_target, A[i], b[i], k[i], m)
        t0   = 0.5 * (hat_t[i] + t_cr)
        if t0 < TSD[i] + 1e-3:
            t0 = TSD[i] + 1e-3
        if t0 > tmax[i]:
            t0 = tmax[i]

        # softplus^-1 approx: s0 = log(exp(x) - 1), x = t0 - TSD
        x = t0 - TSD[i]
        if x > 20.0:
            s_i = x
        elif x < 1e-6:
            s_i = np.log1p(x)
        else:
            s_i = np.log(np.exp(x) - 1.0)

        do_adapt = use_adaptive_variance or (abs(t_cr - hat_t[i]) > 10.0)

        for _ in range(max_iter):
            # reconstruct t and derivatives via chain rule
            sp     = _softplus(s_i)
            dt_ds  = _dsoftplus(s_i)
            ti     = TSD[i] + sp
            if ti > tmax[i]:
                ti = tmax[i]

            Bi   = _cr_forward(ti, A[i], b[i], k[i], m)
            diff = Bi - B_target
            dBdt = _cr_derivative(ti, A[i], b[i], k[i], m)

            # adaptive variance if needed
            if do_adapt:
                dA, dbs, dk_ = _cr_sensitivities_at_t0(ti, A[i], b[i], k[i], m)
                sigma_param2_i = (dA*dA)*(sd_A[i]**2) + (dbs*dbs)*(sd_b[i]**2) + (dk_*dk_)*(sd_k[i]**2)
            else:
                sigma_param2_i = sigma_param2[i]

            sigma_eff2 = sigma_B_meas[i]**2 + sigma_param2_i
            floor2 = (sigma_eff_floor_frac * A[i])**2
            if sigma_eff2 < floor2:
                sigma_eff2 = floor2

            # pseudo-Huber
            delta = robust_delta_frac * A[i]
            if delta < 1.0:
                delta = 1.0
            r = diff / np.sqrt(sigma_eff2)
            w_grad = 1.0 / np.sqrt(1.0 + (r / delta)**2)
            w_hess = w_grad

            # chain rule to s-space
            dB_dsi = dBdt * dt_ds

            # gradient/hessian wrt s (no prior on s → neutral)
            g = ((ti - hat_t[i]) / (sigma_ml[i]**2)) * dt_ds \
                + (diff * dB_dsi) / sigma_eff2 * w_grad

            h = (1.0 / (sigma_ml[i]**2)) * (dt_ds * dt_ds) \
                + (dB_dsi * dB_dsi) / sigma_eff2 * w_hess

            if h < 1e-12:
                break

            step = damping * g / h
            s_new = s_i - step

            if abs(step) < tol:
                s_i = s_new
                break
            s_i = s_new

        # finalize
        t_val = TSD[i] + _softplus(s_i)
        if t_val > tmax[i]:
            t_val = tmax[i]
        t_out[i] = t_val

    return t_out



# ==========================================================
# 4. PUBLIC CLASS
# ==========================================================

class AgeBiasCorrector:
    """
    Ultra-fast Numba-based Bayesian bias corrector for ML forest ages.

    Improvements over the original:
      - Gamma slack prior on (t - TSD) to avoid sticking to the lower bound
      - Optional softplus reparam (no boundary)
      - Robust pseudo-Huber biomass residual
      - Smarter initialization via inverse-CR + slack mode
      - Optional adaptive variance when t moves away from hat_t
      - Small sigma floor to tame overconfident biomass fits

    Public API preserved: precompute_sigma_param2(), correct(), diagnostic_summary()
    """

    def __init__(
        self,
        A, b, k, sd_A, sd_b, sd_k,
        m_fixed=0.67,
        # Slack prior params for t-space solver
        slack_shape=2.0,             # k > 1; mode = (k-1)*theta
        slack_scale=5.0,             # theta (years)
        # Softplus reparam prior on s ~ N(mu_s, sd_s^2)
        softplus_mu_s=5.0,           # years
        softplus_sd_s=5.0,           # years
        # Robust & variance floors
        robust_delta_frac=0.10,      # pseudo-Huber delta ~ 10% of A
        sigma_eff_floor_frac=0.05,   # floor ~ 5% of A
    ):
        self.A    = A.astype(np.float32).reshape(-1)
        self.b    = b.astype(np.float32).reshape(-1)
        self.k    = k.astype(np.float32).reshape(-1)
        self.sd_A = sd_A.astype(np.float32).reshape(-1)
        self.sd_b = sd_b.astype(np.float32).reshape(-1)
        self.sd_k = sd_k.astype(np.float32).reshape(-1)
        self.m    = np.float32(m_fixed)

        # hyper-parameters
        self.slack_shape = np.float32(slack_shape)
        self.slack_scale = np.float32(slack_scale)
        self.softplus_mu_s = np.float32(softplus_mu_s)
        self.softplus_sd_s = np.float32(softplus_sd_s)
        self.robust_delta_frac = np.float32(robust_delta_frac)
        self.sigma_eff_floor_frac = np.float32(sigma_eff_floor_frac)

    def precompute_sigma_param2(self, hat_t):
        hat_t = hat_t.astype(np.float32).reshape(-1)
        return _precompute_sigma_param2(
            self.A, self.b, self.k,
            self.sd_A, self.sd_b, self.sd_k,
            hat_t, self.m
        )

    def correct(
        self, hat_t, B_obs, sigma_ml, sigma_B_meas,
        sigma_param2, TSD, tmax,
        max_iter=20, tol=0.05, damping=0.8,
        adaptive_variance=True,
        use_softplus_reparam=True
    ):
        """
        Bayesian bias correction (MAP).

        Parameters
        ----------
        hat_t : (N,) ML prior mean age
        B_obs : (N,) observed biomass (clamped internally to [0, A])
        sigma_ml : (N,) ML prior std
        sigma_B_meas : (N,) biomass measurement std
        sigma_param2 : (N,) delta-method param variance at hat_t
        TSD : (N,) lower bound array (time-since-disturbance)
        tmax : (N,) hard upper bound array
        max_iter, tol, damping : Newton controls
        adaptive_variance : bool, recompute sigma_param2 per-iteration for movers
        use_softplus_reparam : bool, use t = TSD + softplus(s) instead of slack prior

        Returns
        -------
        (N,) corrected ages
        """
        hat_t        = hat_t.astype(np.float32).reshape(-1)
        B_obs        = B_obs.astype(np.float32).reshape(-1)
        sigma_ml     = sigma_ml.astype(np.float32).reshape(-1)
        sigma_B_meas = sigma_B_meas.astype(np.float32).reshape(-1)
        sigma_param2 = sigma_param2.astype(np.float32).reshape(-1)
        TSD  = np.asarray(TSD,  dtype=np.float32).reshape(-1)
        tmax = np.asarray(tmax, dtype=np.float32).reshape(-1)

        if use_softplus_reparam:
            return _newton_solver_softplus_reparam(
                hat_t, B_obs, sigma_ml, sigma_B_meas,
                self.A, self.b, self.k, sigma_param2,
                TSD, tmax,
                self.m, max_iter, tol, damping,
                self.softplus_mu_s, self.softplus_sd_s,
                self.robust_delta_frac, self.sigma_eff_floor_frac,
                adaptive_variance, self.sd_A, self.sd_b, self.sd_k
            )
        else:
            return _newton_solver_slack_prior(
                hat_t, B_obs, sigma_ml, sigma_B_meas,
                self.A, self.b, self.k, sigma_param2,
                TSD, tmax,
                self.m, max_iter, tol, damping,
                self.slack_shape, self.slack_scale,
                self.robust_delta_frac, self.sigma_eff_floor_frac,
                adaptive_variance, self.sd_A, self.sd_b, self.sd_k
            )

    def diagnostic_summary(self, hat_t, t_corrected, B_obs):
        delta = t_corrected - hat_t
        B_pred = np.empty_like(t_corrected, dtype=np.float32)
        for i in range(len(t_corrected)):
            B_pred[i] = _cr_forward(t_corrected[i], self.A[i], self.b[i], self.k[i], self.m)
        rmse = np.sqrt(np.mean((B_pred - B_obs)**2))

        print("=== Bias Correction Summary ===")
        print(f"Mean correction:   {np.mean(delta):+.2f} years")
        print(f"Median correction: {np.median(delta):+.2f} years")
        print(f"Std correction:    {np.std(delta):.2f} years")
        print(f"Max |correction|:  {np.max(np.abs(delta)):.2f} years")
        print(f"Biomass fit RMSE:  {rmse:.3f}")
