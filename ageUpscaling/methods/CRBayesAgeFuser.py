from __future__ import annotations

# --- imports ---
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from typing import Optional, Tuple

# --- choose backend early ---
try:
    from pytensor.link.numba.dispatch import numba_available
    if numba_available:
        pytensor.config.mode = "NUMBA"
    else:
        pytensor.config.mode = "FAST_RUN"  # fallback if numba missing
except Exception:
    pytensor.config.mode = "FAST_RUN"

class CRBayesAgeFuser:
    """
    Bayesian fusion of ML ages with a Chapman–Richards (CR) growth likelihood in PyMC.

    - Prior:   t ~ TruncatedNormal(mu=hat_t, sigma=sigma_ml, lower=0, upper=tmax)
    - Likelihood: B_obs ~ Normal(mu=B_CR(t; A,k,b), sigma=sigma_B_eff)
      where sigma_B_eff^2 = sigma_B_meas^2 + g(t)^T Σθ g(t)
            g(t) = ∂B/∂(A,k,b) sensitivities (delta method)

    Supports:
      • Diagonal CR-parameter uncertainty via per-parameter stds (sd_A, sd_k, sd_b)
      • OR a full per-pixel covariance Sigma_theta of shape (N,3,3)
      • Soft TSD lower bound via a one-sided quadratic potential
      • Hard upper bound tmax (can be scalar or per-pixel)

    Typical flow:
      fuser = CRBayesAgeFuser(...).build()
      map_res = fuser.find_map()
      # or
      idata = fuser.sample(draws=1000, chains=2)

    You can reuse the model on new tiles/members via `update_data(...)`.
    """

    def __init__(
        self,
        hat_t: np.ndarray,                 # (N,)
        B_obs: np.ndarray,                 # (N,)
        A: np.ndarray, k: np.ndarray, b: np.ndarray,  # (N,)
        sigma_ml: np.ndarray,              # (N,)
        sigma_B_meas: np.ndarray,          # (N,)
        TSD: np.ndarray | float,           # (N,) or scalar
        tmax: np.ndarray | float,          # (N,) or scalar
        sigma_TSD: np.ndarray | float = 8.0,      # (N,) or scalar
        sd_A: Optional[np.ndarray] = None,     # (N,)
        sd_k: Optional[np.ndarray] = None,        # (N,)
        sd_b: Optional[np.ndarray] = None,        # (N,)
        Sigma_theta: Optional[np.ndarray] = None, # (N,3,3) if provided overrides sd_*
        floors: dict = None,               # floors for variances and params
    ):
        self.hat_t  = np.asarray(hat_t,  float).reshape(-1)
        self.B_obs  = np.asarray(B_obs,  float).reshape(-1)
        self.A   = np.asarray(A,   float).reshape(-1)
        self.k      = np.asarray(k,      float).reshape(-1)
        self.b      = np.asarray(b,      float).reshape(-1)
        self.sigma_ml    = np.asarray(sigma_ml,    float).reshape(-1)
        self.sigma_B_meas= np.asarray(sigma_B_meas,float).reshape(-1)

        # broadcast scalars to (N,)
        N = self.hat_t.size
        self.TSD       = self._as_len(self._to_array(TSD), N)
        self.tmax      = self._as_len(self._to_array(tmax), N)
        self.sigma_TSD = self._as_len(self._to_array(sigma_TSD), N)

        # Optional uncertainty spec
        self.Sigma_theta = None
        if Sigma_theta is not None:
            Sigma_theta = np.asarray(Sigma_theta, float)
            assert Sigma_theta.shape == (N, 3, 3), "Sigma_theta must be (N,3,3)"
            self.Sigma_theta = Sigma_theta
        else:
            # Diagonal stds variant
            assert (sd_A is not None) and (sd_k is not None) and (sd_b is not None), \
                "Provide either Sigma_theta (N,3,3) OR per-param stds sd_A, sd_k, sd_b"
            self.sd_A = np.asarray(sd_A, float).reshape(-1)
            self.sd_k    = np.asarray(sd_k,    float).reshape(-1)
            self.sd_b    = np.asarray(sd_b,    float).reshape(-1)

        # Floors
        self.floors = {
            "sigma_ml": 1.0,
            "sigma_B_meas": 1.0,
            "sd_A": 1e-3,
            "sd_k":    1e-5,
            "sd_b":    1e-3,
            "sigma_TSD": 1e-6
        }
        if floors:
            self.floors.update(floors)

        self._sanitize()
        self.model: Optional[pm.Model] = None
        self.vars: dict = {}

    # ---------- utilities ----------

    @staticmethod
    def _to_array(x):
        arr = np.asarray(x, float)
        return arr

    @staticmethod
    def _as_len(x: np.ndarray, N: int) -> np.ndarray:
        if x.ndim == 0:
            return np.full(N, float(x))
        x = x.reshape(-1)
        if x.size != N:
            raise ValueError(f"Expected size {N}, got {x.size}")
        return x

    def _sanitize(self):
        """Clip tiny/invalid sigmas, clip params to reasonable ranges."""
        # floors to avoid zero-variance
        self.sigma_ml     = np.clip(self.sigma_ml,     self.floors["sigma_ml"],     np.inf)
        self.sigma_B_meas = np.clip(self.sigma_B_meas, self.floors["sigma_B_meas"], np.inf)
        self.sigma_TSD    = np.clip(self.sigma_TSD,    self.floors["sigma_TSD"],    np.inf)

        if self.Sigma_theta is None:
            self.sd_A  = np.clip(self.sd_A, self.floors["sd_A"], np.inf)
            self.sd_k     = np.clip(self.sd_k,    self.floors["sd_k"],    np.inf)
            self.sd_b     = np.clip(self.sd_b,    self.floors["sd_b"],    np.inf)

        # keep CR params in sane numeric support (no hard science here, just guards)
        self.A    = np.clip(self.A, 1e-6, np.inf)
        self.k    = np.clip(self.k,    1e-6, 0.5)
        self.b    = np.clip(self.b,    0.2,  8.0)

        # bounds for t
        self.tmax = np.clip(self.tmax, 1.0, np.inf)
        self.hat_t = np.clip(self.hat_t, 0.0, self.tmax)

    # ---------- CR pieces (pytensor) ----------
    
    @staticmethod
    def B_CR_pt(t, A, b, k, m=0.67):
        p = 1.0 / (1.0 - m)
        u = pt.clip(1.0 - b * pt.exp(-k * t), 1e-12, 1 - 1e-12)
        return A * u**p

    @staticmethod
    def CR_sensitivities_pt(t, A, b, k, m=0.67):
        """
        g = [∂y/∂A, ∂y/∂b, ∂y/∂k, ∂y/∂m] for canonical CR.
        If m is fixed, ignore the last column.
        """
        p = 1.0 / (1.0 - m)
        u = pt.clip(1.0 - b * pt.exp(-k * t), 1e-12, 1-1e-12)
        y = A * u**p
    
        # derivatives
        dy_dA = u**p
        dy_db = -A * p * u**(p-1.0) * pt.exp(-k*t)
        dy_dk =  A * p * u**(p-1.0) * (b * t * pt.exp(-k*t))
        # because p = (1-m)^(-1): dp/dm = 1/(1-m)^2
        dy_dm = y * pt.log(u) * (1.0 / (1.0 - m)**2)
    
        return pt.stack([dy_dA, dy_db, dy_dk, dy_dm], axis=-1)

    # ---------- model build ----------

    def build(self) -> "CRBayesAgeFuser":
        """Construct the PyMC model; call once per tile/member (reuse with update_data)."""
        N = self.hat_t.size
        t_init = np.clip(self.hat_t, 1.0, self.tmax)

        with pm.Model() as self.model:
            # constant data containers for fast updates later
            d_hat_t  = pm.Data("hat_t",  self.hat_t)
            d_B_obs  = pm.Data("B_obs",  self.B_obs)
            d_A      = pm.Data("A",   self.A)
            d_k      = pm.Data("k",      self.k)
            d_b      = pm.Data("b",      self.b)
            d_sigML  = pm.Data("sigma_ml",     self.sigma_ml)
            d_sigB   = pm.Data("sigma_B_meas", self.sigma_B_meas)
            d_TSD    = pm.Data("TSD",    self.TSD)
            d_tmax   = pm.Data("tmax",   self.tmax)
            d_sTSD   = pm.Data("sigma_TSD", self.sigma_TSD)

            # Optional parameter uncertainty
            if self.Sigma_theta is None:
                d_sdA = pm.Data("sd_A", self.sd_A)
                d_sdk = pm.Data("sd_k",    self.sd_k)
                d_sdb = pm.Data("sd_b",    self.sd_b)
            else:
                d_Sigma = pm.Data("Sigma_theta", self.Sigma_theta)

            # Latent age per pixel (prior = ML)
            t = pm.TruncatedNormal(
                "t", mu=d_hat_t, sigma=d_sigML, lower=0.0, upper=d_tmax,
                initval=t_init, shape=N
            )

            # Soft TSD potential (one-sided quadratic)
            under = pt.lt(t, d_TSD)
            resid_tsd = (d_TSD - t) * under
            pm.Potential("soft_tsd", -0.5 * pt.sum((resid_tsd / d_sTSD)**2))

            # CR likelihood
            B_pred = self.B_CR_pt(t, d_A, d_b,  d_k)

            if self.Sigma_theta is None:
                g = self.CR_sensitivities_pt(t, d_A, d_b, d_k)
                sigma_param2 = (g[..., 0]**2) * (d_sdA**2) \
                             + (g[..., 1]**2) * (d_sdb**2) \
                             + (g[..., 2]**2) * (d_sdk**2)
            else:
                g = self.CR_sensitivities_pt(t, d_A, d_b, d_k)
                sigma_param2 = pt.sum(pt.matmul(g, d_Sigma) * g, axis=-1)

            sigma_B_eff = pt.sqrt(pt.clip(d_sigB**2 + sigma_param2, 1e-6, 1e12))
            pm.Normal("obs_like", mu=B_pred, sigma=sigma_B_eff, observed=d_B_obs)

            # keep handles
            self.vars = dict(t=t)

        return self

    # ---------- inference helpers ----------

    def find_map(self, start: Optional[dict] = None, **kwargs) -> np.ndarray:
        """Run MAP and return t_MAP (N,)."""
        assert self.model is not None, "Call build() first."
        # sensible defaults
        if start is None:
            start = {
                "t": np.clip(self.hat_t, 0.0, self.tmax)
            }
        with self.model:
            map_res = pm.find_MAP(start=start, **kwargs)
        return map_res["t"]

    def sample(self, draws=1000, chains=2, **kwargs):
        """Run MCMC and return InferenceData. Use for tiles or smaller regions."""
        assert self.model is not None, "Call build() first."
        with self.model:
            idata = pm.sample(draws=draws, chains=chains, init="jitter+adapt_diag",
                              target_accept=0.9, **kwargs)
        return idata

    # ---------- data updating (tiling / members) ----------

    def update_data(
        self,
        hat_t: np.ndarray,
        B_obs: np.ndarray,
        sigma_ml: np.ndarray,
        sigma_B_meas: np.ndarray,
        A: Optional[np.ndarray] = None,
        k: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        TSD: Optional[np.ndarray | float] = None,
        tmax: Optional[np.ndarray | float] = None,
        sigma_TSD: Optional[np.ndarray | float] = None,
        sd_A: Optional[np.ndarray] = None,
        sd_k: Optional[np.ndarray] = None,
        sd_b: Optional[np.ndarray] = None,
        Sigma_theta: Optional[np.ndarray] = None,
    ):
        """
        Update pm.Data containers so you can reuse the compiled graph on a new tile/member.
        All arrays must be shape (N,) matching the original build.
        """
        assert self.model is not None, "Call build() first."
        N = self.hat_t.size
        # reshape + sanitize minimal things
        hat_t  = np.asarray(hat_t,  float).reshape(-1);   assert hat_t.size == N
        B_obs  = np.asarray(B_obs,  float).reshape(-1);   assert B_obs.size == N
        sigma_ml = np.clip(np.asarray(sigma_ml, float).reshape(-1), self.floors["sigma_ml"], np.inf); assert sigma_ml.size == N
        sigma_B_meas = np.clip(np.asarray(sigma_B_meas, float).reshape(-1), self.floors["sigma_B_meas"], np.inf); assert sigma_B_meas.size == N
      
        self.model["hat_t"].set_value(hat_t)
        self.model["B_obs"].set_value(B_obs)
        self.model["sigma_ml"].set_value(sigma_ml)
        self.model["sigma_B_meas"].set_value(sigma_B_meas)

        if A is not None: self.model["A"].set_value(np.asarray(A, float).reshape(-1))
        if k    is not None: self.model["k"].set_value(np.asarray(k, float).reshape(-1))
        if b    is not None: self.model["b"].set_value(np.asarray(b, float).reshape(-1))

        if TSD is not None:       self.model["TSD"].set_value(self._as_len(self._to_array(TSD), N))
        if tmax is not None:      self.model["tmax"].set_value(self._as_len(self._to_array(tmax), N))
        if sigma_TSD is not None: self.model["sigma_TSD"].set_value(np.clip(self._as_len(self._to_array(sigma_TSD), N), self.floors["sigma_TSD"], np.inf))

        # Uncertainty spec
        if Sigma_theta is not None:
            assert Sigma_theta.shape == (N,3,3)
            self.model["Sigma_theta"].set_value(Sigma_theta)
        else:
            if sd_A is not None: self.model["sd_A"].set_value(np.clip(np.asarray(sd_A,float).reshape(-1), self.floors["sd_A"], np.inf))
            if sd_k    is not None: self.model["sd_k"].set_value(np.clip(np.asarray(sd_k,float).reshape(-1),    self.floors["sd_k"],    np.inf))
            if sd_b    is not None: self.model["sd_b"].set_value(np.clip(np.asarray(sd_b,float).reshape(-1),    self.floors["sd_b"],    np.inf))

    # ---------- convenience: run full pipeline once ----------

    @classmethod
    def run_map(
        cls, *args, **kwargs
    ) -> Tuple[np.ndarray, "CRBayesAgeFuser"]:
        """
        Convenience one-shot: build model, run MAP, return (t_MAP, fuser).
        Usage:
            t_map, fuser = CRBayesAgeFuser.run_map(
                hat_t=..., B_obs=..., A=..., k=..., b=...,
                sigma_ml=..., sigma_B_meas=..., TSD=..., tmax=...,
                sd_A=..., sd_k=..., sd_b=...,  # or Sigma_theta=...
            )
        """
        fuser = cls(*args, **kwargs).build()
        t_map = fuser.find_map()
        return t_map, fuser
