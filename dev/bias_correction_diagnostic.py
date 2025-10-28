import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec


def plot_bias_correction_diagnostics(
    idx,
    hat_t, t_corrected, B_obs,
    A, b, k, m_fixed,
    sigma_ml, sigma_B_meas,
    sd_A, sd_b, sd_k,
    TSD=None, tmax=None, sigma_TSD=5.0,
    t_range=None,
    figsize=(14, 5),
    title=None
):
    """
    Beautiful diagnostic plot for a single pixel showing:
    - Growth curve with parameter uncertainty envelope
    - Prior and posterior age distributions
    - Observed biomass with uncertainty
    - TSD constraint visualization
    
    Parameters
    ----------
    idx : int
        Pixel index to plot
    hat_t : array
        Prior age (ML estimate)
    t_corrected : array
        Posterior age (bias-corrected)
    B_obs : array
        Observed biomass
    A, b, k : array
        Chapman-Richards parameters
    m_fixed : float
        Fixed CR shape parameter
    sigma_ml : array
        Prior age uncertainty
    sigma_B_meas : array
        Biomass measurement uncertainty
    sd_A, sd_b, sd_k : array
        CR parameter uncertainties
    TSD : array or None
        Time-since-disturbance lower bound
    tmax : array or None
        Hard upper bound on age
    sigma_TSD : float
        Soft penalty width for TSD
    t_range : tuple or None
        (t_min, t_max) for x-axis. Auto if None.
    figsize : tuple
        Figure size
    title : str or None
        Optional title
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    
    # Extract pixel data
    t_ml = hat_t[idx]
    t_post = t_corrected[idx]
    B = B_obs[idx]
    sig_ml = sigma_ml[idx]
    sig_B = sigma_B_meas[idx]
    
    A_val = A[idx]
    b_val = b[idx]
    k_val = k[idx]
    sdA = sd_A[idx]
    sdb = sd_b[idx]
    sdk = sd_k[idx]
    
    TSD_val = TSD[idx] if TSD is not None else None
    tmax_val = tmax[idx] if tmax is not None else None
    
    # Auto-determine t range
    if t_range is None:
        t_min = max(0, min(t_ml - 3*sig_ml, t_post - 3*sig_ml, TSD_val if TSD_val else 0) - 5)
        t_max = max(t_ml + 3*sig_ml, t_post + 3*sig_ml, tmax_val if tmax_val else 0) + 5
        t_range = (t_min, t_max)
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3, 1], 
                          width_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_age_dist = fig.add_subplot(gs[0, 1])
    ax_residual = fig.add_subplot(gs[1, 0])
    
    # ============================================================
    # MAIN PLOT: Growth Curve + Observations
    # ============================================================
    
    t_vals = np.linspace(t_range[0], t_range[1], 300)
    
    # Mean growth curve
    def cr_forward(t, A, b, k, m):
        eps = 1e-12
        p = 1.0 / (1.0 - m)
        u = np.clip(1.0 - b * np.exp(-k * t), eps, 1.0 - eps)
        return A * (u ** p)
    
    B_mean = cr_forward(t_vals, A_val, b_val, k_val, m_fixed)
    
    # Uncertainty envelope via Monte Carlo sampling
    n_samples = 500
    A_samples = np.random.normal(A_val, sdA, n_samples)
    b_samples = np.random.normal(b_val, sdb, n_samples)
    k_samples = np.random.normal(k_val, sdk, n_samples)
    
    # Clip to valid CR domain
    A_samples = np.clip(A_samples, 1e-6, None)
    b_samples = np.clip(b_samples, 1e-6, 1.0 - 1e-6)
    k_samples = np.clip(k_samples, 1e-6, 1.0)
    
    B_samples = np.array([
        cr_forward(t_vals, A_samples[i], b_samples[i], k_samples[i], m_fixed)
        for i in range(n_samples)
    ])
    
    B_p05 = np.percentile(B_samples, 5, axis=0)
    B_p25 = np.percentile(B_samples, 25, axis=0)
    B_p75 = np.percentile(B_samples, 75, axis=0)
    B_p95 = np.percentile(B_samples, 95, axis=0)
    
    # Plot growth curve with uncertainty
    ax_main.fill_between(t_vals, B_p05, B_p95, alpha=0.15, color='C0', 
                         label='CR 90% CI (param unc.)')
    ax_main.fill_between(t_vals, B_p25, B_p75, alpha=0.25, color='C0', 
                         label='CR 50% CI')
    ax_main.plot(t_vals, B_mean, 'C0-', linewidth=2, label='CR mean')
    
    # Observed biomass with error bar
    ax_main.errorbar(t_post, B, yerr=sig_B, fmt='o', color='darkred', 
                    markersize=10, capsize=5, capthick=2, 
                    label=f'Observed biomass ± σ', zorder=10)
    
    # Prior age (ML) with uncertainty
    B_ml = cr_forward(t_ml, A_val, b_val, k_val, m_fixed)
    ax_main.axvline(t_ml, color='orange', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Prior (ML): {t_ml:.1f} ± {sig_ml:.1f} yr')
    ax_main.axvspan(t_ml - sig_ml, t_ml + sig_ml, alpha=0.1, color='orange')
    ax_main.plot(t_ml, B_ml, 'o', color='orange', markersize=8, alpha=0.7)
    
    # Posterior age (corrected)
    B_post = cr_forward(t_post, A_val, b_val, k_val, m_fixed)
    ax_main.axvline(t_post, color='darkgreen', linestyle='-', linewidth=2.5, 
                   label=f'Posterior (corrected): {t_post:.1f} yr', zorder=5)
    ax_main.plot(t_post, B_post, 's', color='darkgreen', markersize=10, 
                zorder=11, markeredgecolor='white', markeredgewidth=1.5)
    
    # TSD constraint
    if TSD_val is not None:
        ax_main.axvline(TSD_val, color='red', linestyle=':', linewidth=2, 
                       alpha=0.6, label=f'TSD: {TSD_val:.1f} yr')
        ax_main.axvspan(t_range[0], TSD_val, alpha=0.05, color='red', zorder=0)
        
        # Show soft penalty region
        if t_post < TSD_val:
            penalty_region = Rectangle((t_post, 0), TSD_val - t_post, 
                                      ax_main.get_ylim()[1],
                                      alpha=0.1, color='red', zorder=0)
            ax_main.add_patch(penalty_region)
    
    # Tmax constraint
    if tmax_val is not None:
        ax_main.axvline(tmax_val, color='purple', linestyle=':', linewidth=2, 
                       alpha=0.4, label=f'Max age: {tmax_val:.0f} yr')
    
    ax_main.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Biomass (Mg/ha)', fontsize=12, fontweight='bold')
    ax_main.set_xlim(t_range)
    ax_main.grid(alpha=0.3, linestyle='--')
    ax_main.legend(loc='best', fontsize=9, framealpha=0.95)
    
    # ============================================================
    # AGE DISTRIBUTION PLOT (right panel)
    # ============================================================
    
    age_vals = np.linspace(t_range[0], t_range[1], 200)
    
    # Prior distribution
    prior_dist = np.exp(-0.5 * ((age_vals - t_ml) / sig_ml) ** 2)
    prior_dist /= prior_dist.max()
    
    ax_age_dist.plot(prior_dist, age_vals, 'orange', linewidth=2, 
                    label='Prior (ML)')
    ax_age_dist.fill_betweenx(age_vals, 0, prior_dist, alpha=0.2, color='orange')
    
    # Posterior (approximated as delta function, or could compute Laplace approx)
    # For visualization, show as narrow Gaussian
    # Compute posterior uncertainty via Laplace approximation
    def compute_posterior_std(t, A, b, k, m, sig_ml, sig_B, sdA, sdb, sdk):
        eps = 1e-12
        p = 1.0 / (1.0 - m)
        u = np.clip(1.0 - b * np.exp(-k * t), eps, 1.0 - eps)
        
        # dB/dt
        dBdt = A * p * (u ** (p - 1.0)) * (b * k * np.exp(-k * t))
        
        # Parameter sensitivities
        gA = u ** p
        gb = -A * p * (u ** (p - 1.0)) * np.exp(-k * t)
        gk = A * p * (u ** (p - 1.0)) * (b * t * np.exp(-k * t))
        sigma_param2 = (gA**2)*(sdA**2) + (gb**2)*(sdb**2) + (gk**2)*(sdk**2)
        
        # Hessian diagonal
        H = 1.0 / (sig_ml**2) + (dBdt**2) / (sig_B**2 + sigma_param2)
        return 1.0 / np.sqrt(H)
    
    sig_post = compute_posterior_std(t_post, A_val, b_val, k_val, m_fixed, 
                                     sig_ml, sig_B, sdA, sdb, sdk)
    
    post_dist = np.exp(-0.5 * ((age_vals - t_post) / sig_post) ** 2)
    post_dist /= post_dist.max()
    
    ax_age_dist.plot(post_dist, age_vals, 'darkgreen', linewidth=2.5, 
                    label='Posterior')
    ax_age_dist.fill_betweenx(age_vals, 0, post_dist, alpha=0.25, color='darkgreen')
    
    # Mark actual values
    ax_age_dist.axhline(t_ml, color='orange', linestyle='--', alpha=0.5)
    ax_age_dist.axhline(t_post, color='darkgreen', linestyle='-', linewidth=2, alpha=0.8)
    
    if TSD_val is not None:
        ax_age_dist.axhline(TSD_val, color='red', linestyle=':', linewidth=2, alpha=0.6)
        ax_age_dist.axhspan(t_range[0], TSD_val, alpha=0.05, color='red')
    
    ax_age_dist.set_ylim(t_range)
    ax_age_dist.set_xlabel('Probability\n(normalized)', fontsize=10, fontweight='bold')
    ax_age_dist.set_ylabel('')
    ax_age_dist.set_xlim([0, 1.1])
    ax_age_dist.set_xticks([0, 0.5, 1])
    ax_age_dist.legend(loc='upper right', fontsize=9)
    ax_age_dist.grid(alpha=0.3, axis='y', linestyle='--')
    ax_age_dist.yaxis.tick_right()
    
    # ============================================================
    # RESIDUAL PLOT (bottom left)
    # ============================================================
    
    # Predicted biomass along growth curve
    B_curve = cr_forward(t_vals, A_val, b_val, k_val, m_fixed)
    residuals = B - B_curve
    
    # Normalize by uncertainty
    sigma_eff = np.sqrt(sig_B**2 + (B_p75 - B_p25)**2 / 2)  # rough estimate
    normalized_residuals = residuals / sigma_eff
    
    # Color by magnitude
    colors = plt.cm.RdYlGn_r(np.clip(np.abs(normalized_residuals) / 3, 0, 1))
    
    for i in range(len(t_vals) - 1):
        ax_residual.plot(t_vals[i:i+2], residuals[i:i+2], 
                        color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax_residual.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_residual.axhline(sig_B, color='gray', linestyle='--', alpha=0.5, 
                       label=f'±σ_B ({sig_B:.1f})')
    ax_residual.axhline(-sig_B, color='gray', linestyle='--', alpha=0.5)
    
    # Mark prior and posterior
    resid_ml = B - cr_forward(t_ml, A_val, b_val, k_val, m_fixed)
    resid_post = B - cr_forward(t_post, A_val, b_val, k_val, m_fixed)
    
    ax_residual.plot(t_ml, resid_ml, 'o', color='orange', markersize=8, 
                    label=f'Prior residual: {resid_ml:.1f}')
    ax_residual.plot(t_post, resid_post, 's', color='darkgreen', markersize=10, 
                    label=f'Posterior residual: {resid_post:.1f}',
                    markeredgecolor='white', markeredgewidth=1.5)
    
    if TSD_val is not None:
        ax_residual.axvline(TSD_val, color='red', linestyle=':', linewidth=2, alpha=0.4)
    
    ax_residual.set_xlabel('Age (years)', fontsize=11, fontweight='bold')
    ax_residual.set_ylabel('Residual (Mg/ha)', fontsize=11, fontweight='bold')
    ax_residual.set_xlim(t_range)
    ax_residual.grid(alpha=0.3, linestyle='--')
    ax_residual.legend(loc='best', fontsize=9)
    
    # ============================================================
    # Title and annotations
    # ============================================================
    
    if title is None:
        correction = t_post - t_ml
        title = (f'Pixel {idx}: Correction = {correction:+.1f} years  '
                f'|  Prior σ = {sig_ml:.1f} yr  |  Post σ = {sig_post:.1f} yr')
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Add metadata box
    metadata_text = (
        f'CR params: A={A_val:.1f}±{sdA:.1f}, b={b_val:.3f}±{sdb:.3f}, '
        f'k={k_val:.4f}±{sdk:.4f}\n'
        f'Obs: B={B:.1f}±{sig_B:.1f} Mg/ha'
    )
    
    ax_residual.text(0.02, 0.02, metadata_text, transform=ax_residual.transAxes,
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    return fig, [ax_main, ax_age_dist, ax_residual]


def plot_correction_summary(
    hat_t, t_corrected, B_obs,
    A, b, k, m_fixed,
    sigma_ml, sigma_B_meas,
    TSD=None,
    sample_size=9,
    figsize=(18, 12),
    seed=42
):
    """
    Multi-panel summary showing representative pixels.
    
    Parameters
    ----------
    sample_size : int
        Number of pixels to show (arranged in grid)
    seed : int
        Random seed for sampling pixels
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    np.random.seed(seed)
    N = len(hat_t)
    
    # Sample diverse pixels (large corrections, small corrections, etc.)
    corrections = t_corrected - hat_t
    
    # Stratified sampling
    large_pos = np.where(corrections > np.percentile(corrections, 90))[0]
    large_neg = np.where(corrections < np.percentile(corrections, 10))[0]
    medium = np.where(np.abs(corrections) < np.percentile(np.abs(corrections), 50))[0]
    
    n_per_group = sample_size // 3
    indices = np.concatenate([
        np.random.choice(large_pos, min(n_per_group, len(large_pos)), replace=False),
        np.random.choice(large_neg, min(n_per_group, len(large_neg)), replace=False),
        np.random.choice(medium, sample_size - 2*n_per_group, replace=False)
    ])
    
    # Create grid
    nrows = int(np.sqrt(sample_size))
    ncols = int(np.ceil(sample_size / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Simplified single-panel version
        t_ml = hat_t[idx]
        t_post = t_corrected[idx]
        B = B_obs[idx]
        sig_ml = sigma_ml[idx]
        sig_B = sigma_B_meas[idx]
        
        # Growth curve
        t_vals = np.linspace(max(0, t_ml - 3*sig_ml - 10), 
                            t_ml + 3*sig_ml + 10, 200)
        
        def cr_forward(t, A, b, k, m):
            eps = 1e-12
            p = 1.0 / (1.0 - m)
            u = np.clip(1.0 - b * np.exp(-k * t), eps, 1.0 - eps)
            return A * (u ** p)
        
        B_curve = cr_forward(t_vals, A[idx], b[idx], k[idx], m_fixed)
        
        ax.plot(t_vals, B_curve, 'C0-', linewidth=1.5, alpha=0.7)
        ax.errorbar(t_post, B, yerr=sig_B, fmt='o', color='darkred', 
                   markersize=6, capsize=3)
        ax.axvline(t_ml, color='orange', linestyle='--', alpha=0.6)
        ax.axvline(t_post, color='darkgreen', linestyle='-', linewidth=2)
        
        if TSD is not None:
            ax.axvline(TSD[idx], color='red', linestyle=':', alpha=0.5)
        
        correction = t_post - t_ml
        ax.set_title(f'Pixel {idx}: {correction:+.1f} yr', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel('Age (yr)', fontsize=8)
        ax.set_ylabel('Biomass', fontsize=8)
    
    # Remove empty subplots
    for i in range(len(indices), len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle('Bias Correction Summary: Representative Pixels', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


    
# Generate synthetic data
np.random.seed(42)
N = 100

hat_t = np.random.uniform(20, 80, N)
sigma_ml = np.random.uniform(5, 15, N)

A = np.random.uniform(150, 250, N)
b = np.random.uniform(0.6, 0.9, N)
k = np.random.uniform(0.02, 0.06, N)
m_fixed = 0.67

sd_A = 0.1 * A
sd_b = 0.05 * b
sd_k = 0.01 * k

# Simulate observations
def cr_forward(t, A, b, k, m):
    p = 1.0 / (1.0 - m)
    u = np.clip(1.0 - b * np.exp(-k * t), 1e-12, 1.0 - 1e-12)
    return A * (u ** p)
    
B_true = cr_forward(hat_t, A, b, k, m_fixed)
B_obs = B_true + np.random.normal(0, 15, N)
sigma_B_meas = np.full(N, 15.0)

TSD = np.random.uniform(5, 30, N)
tmax = np.full(N, 150.0)

# Simulate corrections (in real use, call corrector.correct())
t_corrected = hat_t + np.random.normal(-5, 8, N)
t_corrected = np.clip(t_corrected, TSD, tmax)


# Single pixel detailed view
idx = 42  # interesting pixel
fig, axes = plot_bias_correction_diagnostics(
    idx=idx,
    hat_t=hat_t, 
    t_corrected=t_corrected, 
    B_obs=B_obs,
    A=A, b=b, k=k, 
    m_fixed=0.67,
    sigma_ml=sigma_ml, 
    sigma_B_meas=sigma_B_meas,
    sd_A=sd_A, sd_b=sd_b, sd_k=sd_k,
    TSD=TSD, 
    tmax=tmax,
    sigma_TSD=5.0
)
plt.savefig(f'pixel_{idx}_diagnostic.png', dpi=300, bbox_inches='tight')    