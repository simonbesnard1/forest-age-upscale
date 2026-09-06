import numpy as np
from numba import njit, prange

# ==========================================================
# 0. NUMERICAL HELPERS
# ==========================================================

@njit(cache=True, fastmath=True, parallel=False)
def _softplus(x):
    # stable softplus approximation for t = TSD + softplus(s)
    if x > 20.0:
        return x
    if x < -20.0:
        return np.exp(x)
    return np.log1p(np.exp(x))

@njit(cache=True, fastmath=True, parallel=False)
def _dsoftplus(x):
    # derivative of softplus = sigmoid(x)
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return np.exp(x)
    e = np.exp(x)
    return e / (1.0 + e)


# ==========================================================
# 1. CHAPMAN–RICHARDS
#    B = A * (1 - b * exp(-k t))^(1/(1-m))
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
    Invert CR for t given B (clamped).
    t = -(1/k) * ln( (1 - (B/A)^(1-m)) / b )
    """
    eps = 1e-12

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

    q = 1.0 - r ** (1.0 - m)
    z = q / b_
    if z < eps:
        z = eps
    if z > 1.0 - eps:
        z = 1.0 - eps

    return -np.log(z) / k_


# ==========================================================
# 2. STATIC SIGMA_PARAM² (delta method, precomputed)
#    Full delta-method propagation through the CR parameter
#    covariance: Var(B) = J Sigma J^T, i.e.
#      Var(B) = dA^2 Var(A) + db^2 Var(b) + dk^2 Var(k)
#              + 2 dA db Cov(A,b) + 2 dA dk Cov(A,k) + 2 db dk Cov(b,k)
#    with Cov(X,Y) = rho_XY * sd_X * sd_Y. A, b, k come from the same
#    nonlinear curve fit and are generally correlated, so the cross
#    terms are not optional -- omitting them (rho=0) silently assumes
#    independence that the fit does not have.
# ==========================================================

@njit(cache=True, fastmath=True, parallel=True)
def _precompute_sigma_param2(A, b, k, sd_A, sd_b, sd_k, rho_Ab, rho_Ak, rho_bk, hat_t, m):
    N = A.size
    out = np.empty(N, dtype=np.float32)
    for i in prange(N):
        dA, db, dk = _cr_sensitivities_at_t0(hat_t[i], A[i], b[i], k[i], m)
        var = (dA*dA)*(sd_A[i]**2) + (db*db)*(sd_b[i]**2) + (dk*dk)*(sd_k[i]**2)
        cov = (2.0*dA*db*rho_Ab[i]*sd_A[i]*sd_b[i]
               + 2.0*dA*dk*rho_Ak[i]*sd_A[i]*sd_k[i]
               + 2.0*db*dk*rho_bk[i]*sd_b[i]*sd_k[i])
        total = var + cov
        if total < 0.0:
            total = 0.0
        out[i] = total
    return out


# ==========================================================
# 3. NEWTON SOLVERS
#    TSD used as a hard lower bound ONLY.
#    No prior pulling towards TSD.
#    If solution hugs TSD, fall back to ML.
# ==========================================================

@njit(cache=True, fastmath=True, parallel=True)
def _newton_solver_tspace(
    hat_t, B_obs, sigma_ml, sigma_B_meas,
    A, b, k, sigma_param2, TSD, tmax,
    m, max_iter, tol, damping,
    robust_delta_frac, sigma_eff_floor_frac,
    use_adaptive_variance, sd_A, sd_b, sd_k,
    rho_Ab, rho_Ak, rho_bk,
    bound_fallback_eps
):
    N = hat_t.size
    t_out = np.empty(N, dtype=np.float32)

    for i in prange(N):
        # Clamp biomass to [0, A]
        B_target = B_obs[i]
        if B_target < 0.0:
            B_target = 0.0
        if B_target > A[i]:
            B_target = A[i]

        # Init from inverse-CR & ML blend within bounds
        t_cr = _cr_inverse(B_target, A[i], b[i], k[i], m)
        ti = 0.5 * (hat_t[i] + t_cr)
        
        if ti < TSD[i] + 1e-3:
            ti = TSD[i] + 1e-3
        if ti > tmax[i]:
            ti = tmax[i]

        do_adapt = use_adaptive_variance or (abs(t_cr - hat_t[i]) > 10.0)

        for _ in range(max_iter):
            # Param uncertainty
            if do_adapt:
                dA, dbs, dk_ = _cr_sensitivities_at_t0(ti, A[i], b[i], k[i], m)
                var_i = (dA*dA)*(sd_A[i]**2) + (dbs*dbs)*(sd_b[i]**2) + (dk_*dk_)*(sd_k[i]**2)
                cov_i = (2.0*dA*dbs*rho_Ab[i]*sd_A[i]*sd_b[i]
                         + 2.0*dA*dk_*rho_Ak[i]*sd_A[i]*sd_k[i]
                         + 2.0*dbs*dk_*rho_bk[i]*sd_b[i]*sd_k[i])
                sigma_param2_i = var_i + cov_i
                if sigma_param2_i < 0.0:
                    sigma_param2_i = 0.0
            else:
                sigma_param2_i = sigma_param2[i]

            # Forward + derivative
            Bi   = _cr_forward(ti, A[i], b[i], k[i], m)
            diff = Bi - B_target
            dBdt = _cr_derivative(ti, A[i], b[i], k[i], m)

            # Effective variance
            sigma_eff2 = sigma_B_meas[i]**2 + sigma_param2_i
            floor2 = (sigma_eff_floor_frac * A[i])**2
            if sigma_eff2 < floor2:
                sigma_eff2 = floor2

            # Robust pseudo-Huber
            delta = robust_delta_frac * A[i]
            if delta < 1.0:
                delta = 1.0
            r = diff / np.sqrt(sigma_eff2)
            w_grad = 1.0 / np.sqrt(1.0 + (r/delta)**2)
            w_hess = w_grad

            # Gradient / Hessian of -log posterior wrt t
            g = (ti - hat_t[i]) / (sigma_ml[i]**2) \
                + (diff * dBdt) / sigma_eff2 * w_grad

            h = 1.0 / (sigma_ml[i]**2) \
                + (dBdt*dBdt) / sigma_eff2 * w_hess

            if h < 1e-12:
                break

            step = damping * g / h
            ti_new = ti - step

            # Enforce hard bounds
            if ti_new < TSD[i] + 1e-6:
                ti_new = TSD[i] + 1e-6
            if ti_new > tmax[i]:
                ti_new = tmax[i]

            if abs(step) < tol:
                ti = ti_new
                break
            ti = ti_new

        # Final clamp
        if ti < TSD[i]:
            ti = TSD[i]
        if ti > tmax[i]:
            ti = tmax[i]

        # If glued to lower bound -> keep ML (clipped)
        if ti <= TSD[i] + bound_fallback_eps:
            ti = hat_t[i]
            if ti < TSD[i]:
                ti = TSD[i]
            if ti > tmax[i]:
                ti = tmax[i]

        t_out[i] = ti

    return t_out


@njit(cache=True, fastmath=True, parallel=True)
def _newton_solver_softplus(
    hat_t, B_obs, sigma_ml, sigma_B_meas,
    A, b, k, sigma_param2, TSD, tmax,
    m, max_iter, tol, damping,
    robust_delta_frac, sigma_eff_floor_frac,
    use_adaptive_variance, sd_A, sd_b, sd_k,
    rho_Ab, rho_Ak, rho_bk,
    bound_fallback_eps
):
    """
    Same objective as t-space solver, but parameterized as:
        t = TSD + softplus(s)
    to guarantee t >= TSD. No prior pulling toward TSD.
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

        # Init from inverse-CR & ML blend within bounds
        t_cr = _cr_inverse(B_target, A[i], b[i], k[i], m)
        t0 = 0.5 * (hat_t[i] + t_cr)
        if t0 < TSD[i] + 1e-3:
            t0 = TSD[i] + 1e-3
        if t0 > tmax[i]:
            t0 = tmax[i]

        # softplus^-1 approx: solve for s0 s.t. softplus(s0) ≈ t0 - TSD
        x = t0 - TSD[i]
        if x > 20.0:
            s_i = x
        elif x < 1e-6:
            s_i = np.log1p(x)
        else:
            s_i = np.log(np.exp(x) - 1.0)

        do_adapt = use_adaptive_variance or (abs(t_cr - hat_t[i]) > 10.0)

        for _ in range(max_iter):
            sp    = _softplus(s_i)
            dt_ds = _dsoftplus(s_i)
            ti    = TSD[i] + sp
            if ti > tmax[i]:
                ti = tmax[i]
                # t is pinned at tmax regardless of s beyond this point,
                # so dt/ds is actually 0 here -- using the unclipped
                # softplus derivative would tell the optimizer that
                # increasing s still moves t, which is false.
                dt_ds = 0.0

            Bi   = _cr_forward(ti, A[i], b[i], k[i], m)
            diff = Bi - B_target
            dBdt = _cr_derivative(ti, A[i], b[i], k[i], m)

            # Param variance
            if do_adapt:
                dA, dbs, dk_ = _cr_sensitivities_at_t0(ti, A[i], b[i], k[i], m)
                var_i = (dA*dA)*(sd_A[i]**2) + (dbs*dbs)*(sd_b[i]**2) + (dk_*dk_)*(sd_k[i]**2)
                cov_i = (2.0*dA*dbs*rho_Ab[i]*sd_A[i]*sd_b[i]
                         + 2.0*dA*dk_*rho_Ak[i]*sd_A[i]*sd_k[i]
                         + 2.0*dbs*dk_*rho_bk[i]*sd_b[i]*sd_k[i])
                sigma_param2_i = var_i + cov_i
                if sigma_param2_i < 0.0:
                    sigma_param2_i = 0.0
            else:
                sigma_param2_i = sigma_param2[i]

            sigma_eff2 = sigma_B_meas[i]**2 + sigma_param2_i
            floor2 = (sigma_eff_floor_frac * A[i])**2
            if sigma_eff2 < floor2:
                sigma_eff2 = floor2

            # robust
            delta = robust_delta_frac * A[i]
            if delta < 1.0:
                delta = 1.0
            r = diff / np.sqrt(sigma_eff2)
            w_grad = 1.0 / np.sqrt(1.0 + (r/delta)**2)
            w_hess = w_grad

            dB_ds = dBdt * dt_ds

            # Gradient / Hessian wrt s (chain rule)
            g = ((ti - hat_t[i]) / (sigma_ml[i]**2)) * dt_ds \
                + (diff * dB_ds) / sigma_eff2 * w_grad

            h = (1.0 / (sigma_ml[i]**2)) * (dt_ds * dt_ds) \
                + (dB_ds * dB_ds) / sigma_eff2 * w_hess

            if h < 1e-12:
                break

            step = damping * g / h
            s_new = s_i - step

            if abs(step) < tol:
                s_i = s_new
                break
            s_i = s_new

        # Final t
        t_val = TSD[i] + _softplus(s_i)
        if t_val < TSD[i]:
            t_val = TSD[i]
        if t_val > tmax[i]:
            t_val = tmax[i]

        # If glued to lower bound -> keep ML (clipped)
        if t_val <= TSD[i] + bound_fallback_eps:
            t_val = hat_t[i]
            if t_val < TSD[i]:
                t_val = TSD[i]
            if t_val > tmax[i]:
                t_val = tmax[i]

        t_out[i] = t_val

    return t_out


# ==========================================================
# 4. PUBLIC CLASS
# ==========================================================

class AgeBiasCorrector:
    """
    Bayesian-ish bias corrector for ML forest ages.

    - Uses ML age as Gaussian prior.
    - Uses CR curve + biomass with robust loss.
    - TSD and tmax are *hard bounds only* (no shrinkage toward TSD).
    - If the MAP sticks to TSD, we fall back to the bounded ML age
      to avoid artificial spikes at the lower bound.

    Parameter-uncertainty propagation (rho_Ab, rho_Ak, rho_bk):
    A, b, k are fit jointly by the same nonlinear regression (Robinson
    et al.) and are generally correlated -- treating them as independent
    (rho=0) is an approximation, not a neutral default. The source cube
    (CR_growth_curve_params_1km) only ships marginal standard errors
    (A_error/b_error/k_error), no covariance/correlation terms, so
    rho_Ab/rho_Ak/rho_bk default to 0.0 here purely for lack of data.
    Pass real correlation coefficients (scalar or per-pixel array) if/when
    a covariance matrix becomes available (e.g. by refitting the curves
    with a nonlinear least-squares routine that reports one).
    """

    def __init__(
        self,
        A, b, k, sd_A, sd_b, sd_k,
        m_fixed=0.67,
        robust_delta_frac=0.05,
        sigma_eff_floor_frac=0.15,
        bound_fallback_eps=0.5,   # <= this near TSD → revert to ML
        rho_Ab=0.0, rho_Ak=0.0, rho_bk=0.0,
    ):
        self.A    = A.astype(np.float32).reshape(-1)
        self.b    = b.astype(np.float32).reshape(-1)
        self.k    = k.astype(np.float32).reshape(-1)
        self.sd_A = sd_A.astype(np.float32).reshape(-1)
        self.sd_b = sd_b.astype(np.float32).reshape(-1)
        self.sd_k = sd_k.astype(np.float32).reshape(-1)
        self.m    = np.float32(m_fixed)

        self.robust_delta_frac = np.float32(robust_delta_frac)
        self.sigma_eff_floor_frac = np.float32(sigma_eff_floor_frac)
        self.bound_fallback_eps = np.float32(bound_fallback_eps)

        n = self.A.size
        self.rho_Ab = np.broadcast_to(np.float32(rho_Ab), (n,)).astype(np.float32).copy()
        self.rho_Ak = np.broadcast_to(np.float32(rho_Ak), (n,)).astype(np.float32).copy()
        self.rho_bk = np.broadcast_to(np.float32(rho_bk), (n,)).astype(np.float32).copy()

    def precompute_sigma_param2(self, hat_t):
        hat_t = hat_t.astype(np.float32).reshape(-1)
        return _precompute_sigma_param2(
            self.A, self.b, self.k,
            self.sd_A, self.sd_b, self.sd_k,
            self.rho_Ab, self.rho_Ak, self.rho_bk,
            hat_t, self.m
        )

    def correct(
        self, hat_t, B_obs, sigma_ml, sigma_B_meas,
        sigma_param2, TSD, tmax,
        max_iter=20, tol=0.05, damping=0.8,
        adaptive_variance=True,
        use_softplus_reparam=True,
    ):
        # flatten & cast
        hat_t        = hat_t.astype(np.float32).reshape(-1)
        B_obs        = B_obs.astype(np.float32).reshape(-1)
        sigma_ml     = sigma_ml.astype(np.float32).reshape(-1)
        sigma_B_meas = sigma_B_meas.astype(np.float32).reshape(-1)
        sigma_param2 = sigma_param2.astype(np.float32).reshape(-1)
        TSD          = np.asarray(TSD,  dtype=np.float32).reshape(-1)
        tmax         = np.asarray(tmax, dtype=np.float32).reshape(-1)

        # Ensure prior respects bounds
        for i in range(hat_t.size):
            if hat_t[i] < TSD[i]:
                hat_t[i] = TSD[i]
            if hat_t[i] > tmax[i]:
                hat_t[i] = tmax[i]

        if use_softplus_reparam:
            return _newton_solver_softplus(
                hat_t, B_obs, sigma_ml, sigma_B_meas,
                self.A, self.b, self.k, sigma_param2,
                TSD, tmax,
                self.m, max_iter, tol, damping,
                self.robust_delta_frac, self.sigma_eff_floor_frac,
                adaptive_variance, self.sd_A, self.sd_b, self.sd_k,
                self.rho_Ab, self.rho_Ak, self.rho_bk,
                self.bound_fallback_eps
            )
        else:
            return _newton_solver_tspace(
                hat_t, B_obs, sigma_ml, sigma_B_meas,
                self.A, self.b, self.k, sigma_param2,
                TSD, tmax,
                self.m, max_iter, tol, damping,
                self.robust_delta_frac, self.sigma_eff_floor_frac,
                adaptive_variance, self.sd_A, self.sd_b, self.sd_k,
                self.rho_Ab, self.rho_Ak, self.rho_bk,
                self.bound_fallback_eps
            )
