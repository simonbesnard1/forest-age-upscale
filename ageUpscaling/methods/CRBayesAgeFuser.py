import numpy as np
from numba import njit, prange


# ==========================================================
# 1. CR FORWARD MODEL (Numba)
# ==========================================================

@njit(cache=True, fastmath=True, parallel=False)
def _cr_forward(t, A, b, k, m):
    """
    Compute B_CR(t) = A * (1 - b * exp(-k t))^(1/(1-m))
    Fastmath allowed (small fp diffs, ok for bias correction).
    """
    eps = 1e-12
    p = 1.0 / (1.0 - m)
    u = 1.0 - b * np.exp(-k * t)
    if u < eps:
        u = eps
    if u > 1.0 - eps:
        u = 1.0 - eps
    return A * (u ** p)


@njit(cache=True, fastmath=True, parallel=False)
def _cr_sensitivities_at_t0(t0, A, b, k, m):
    """
    Compute sensitivities [dB/dA, dB/db, dB/dk] at t0.
    These are only used ONCE to build static sigma_param2.
    """
    eps = 1e-12
    p = 1.0 / (1.0 - m)
    u = 1.0 - b * np.exp(-k * t0)
    if u < eps:
        u = eps
    if u > 1.0 - eps:
        u = 1.0 - eps
    # dB/dA:
    dA = u ** p
    # dB/db:
    db = -A * p * (u ** (p - 1.0)) * np.exp(-k * t0)
    # dB/dk:
    dk = A * p * (u ** (p - 1.0)) * (b * t0 * np.exp(-k * t0))
    return dA, db, dk


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
# 3. NEWTON SOLVER (MAP) - PARALLEL
# ==========================================================

@njit(cache=True, fastmath=True, parallel=True)
def _newton_solver(hat_t, B_obs, sigma_ml, sigma_B_meas,
                   A, b, k, sigma_param2,
                   TSD, sigma_TSD, tmax,
                   m, max_iter, tol):
    N = hat_t.size
    t = hat_t.copy()  # initialization

    for i in prange(N):
        ti = t[i]

        # Iterate Newton updates
        for _ in range(max_iter):
            # Forward CR
            Bi = _cr_forward(ti, A[i], b[i], k[i], m)

            # adaptive clamp of B_obs to valid support
            B_target = B_obs[i]
            if B_target > A[i]:
                B_target = A[i]
            if B_target < 0.0:
                B_target = 0.0

            # Residual
            diff = Bi - B_target

            # Soft TSD gradient & Hessian
            under = (ti < TSD[i])
            soft_g = 0.0
            soft_h = 0.0
            if under:
                d = (TSD[i] - ti)
                soft_g = -(d) / (sigma_TSD[i]**2)
                soft_h =  1.0 / (sigma_TSD[i]**2)

            # Derivative of CR wrt t:
            # Use chain rule on forward eq
            eps = 1e-12
            p  = 1.0 / (1.0 - m)
            u  = 1.0 - b[i]*np.exp(-k[i]*ti)
            if u < eps:  u = eps
            if u > 1.0-eps: u = 1.0-eps
            # B = A * u^p, so dB/dt = A * p * u^(p-1) * (b*k*exp(-k*t))
            dBdt = A[i] * p * (u**(p-1.0)) * (b[i]*k[i]*np.exp(-k[i]*ti))

            # Gradient of posterior wrt t
            g = (ti - hat_t[i]) / (sigma_ml[i]**2) \
                + (diff * dBdt) / (sigma_B_meas[i]**2 + sigma_param2[i]) \
                + soft_g

            # Hessian (lag approximation -> stable)
            h = (1.0 / (sigma_ml[i]**2)) \
                + (dBdt*dBdt) / (sigma_B_meas[i]**2 + sigma_param2[i]) \
                + soft_h

            if h <= 0.0:
                break

            step = g / h
            ti_new = ti - step

            # hard bounds
            if ti_new > tmax[i]:
                ti_new = tmax[i]
            if ti_new < 0:
                ti_new = 0.0

            # convergence
            if abs(step) < tol:
                ti = ti_new
                break
            ti = ti_new

        t[i] = ti

    return t


# ==========================================================
# 4. PUBLIC CLASS INTERFACE
# ==========================================================

class AgeBiasCorrector:
    def __init__(self, A, b, k, sd_A, sd_b, sd_k, m_fixed=0.67, sigma_TSD=5.0):
        self.A     = A.astype(np.float32).reshape(-1)
        self.b     = b.astype(np.float32).reshape(-1)
        self.k     = k.astype(np.float32).reshape(-1)
        self.sd_A  = sd_A.astype(np.float32).reshape(-1)
        self.sd_b  = sd_b.astype(np.float32).reshape(-1)
        self.sd_k  = sd_k.astype(np.float32).reshape(-1)
        self.m     = np.float32(m_fixed)   # fixed m from dataset
        self.sigma_TSD_default = np.float32(sigma_TSD)

    def precompute_sigma_param2(self, hat_t):
        hat_t = hat_t.astype(np.float32).reshape(-1)
        return _precompute_sigma_param2(self.A, self.b, self.k,
                                        self.sd_A, self.sd_b, self.sd_k,
                                        hat_t, self.m)

    def correct(self, hat_t, B_obs, sigma_ml, sigma_B_meas,
                sigma_param2, TSD, tmax,
                max_iter=15, tol=1e-3):

        hat_t        = hat_t.astype(np.float32).reshape(-1)
        B_obs        = B_obs.astype(np.float32).reshape(-1)
        sigma_ml     = sigma_ml.astype(np.float32).reshape(-1)
        sigma_B_meas = sigma_B_meas.astype(np.float32).reshape(-1)
        sigma_param2 = sigma_param2.astype(np.float32).reshape(-1)

        TSD  = np.asarray(TSD,  dtype=np.float32).reshape(-1)
        tmax = np.asarray(tmax, dtype=np.float32).reshape(-1)
        sigma_TSD = np.ones_like(hat_t) * self.sigma_TSD_default

        return _newton_solver(
            hat_t, B_obs,
            sigma_ml, sigma_B_meas,
            self.A, self.b, self.k,
            sigma_param2,
            TSD, sigma_TSD, tmax,
            self.m,
            max_iter, tol
        )
