import numpy as np
import yaml as yml
import pickle

from rasterio.features import geometry_mask

import xarray as xr
import dask.array as da
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
import xgboost as xgb

from ageUpscaling.transformers.spatial import interpolate_worlClim
from ageUpscaling.methods.AgeFusion import AgeFusion

study_dir = '/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-3.0'

with open('/home/simon/hpc_home/projects/forest-age-upscale/config_files/upscaling/100m/data_config_xgboost.yaml', 'r') as f:
    DataConfig =  yml.safe_load(f)

with open('/home/simon/hpc_home/projects/forest-age-upscale/config_files/upscaling/100m/config_upscaling.yaml', 'r') as f:
    upscaling_config =  yml.safe_load(f)

LastTimeSinceDist_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/LandsatDisturbanceTime_100m')
agb_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v6_members/')
clim_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/WorlClim_1km')
canopyHeight_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/canopyHeight_potapov_100m')
CRparams_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/CR_growth_curve_params_1km')
agb_std_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v6')

intact_forest = gpd.read_file('/home/simon/hpc_home/projects/forest-age-upscale/data/shapefiles/IFL_2020_v2.zip')
intact_tropical_forest = intact_forest[intact_forest['IFL_ID'].str.contains('|'.join(['SAM', 'SEA', 'AFR']))]

algorithm = "XGBoost"
IN = {'longitude': slice(10.416508, 10.487892 , None) , 'latitude': slice(51.101666, 51.056723 , None) }
#IN = {'latitude': slice(-2.91, -3.118, None) , 'longitude': slice(-55.07 , -54.8714,  None) }

lat_start, lat_stop = IN['latitude'].start, IN['latitude'].stop
lon_start, lon_stop = IN['longitude'].start, IN['longitude'].stop
buffer_IN = Polygon([(lon_start, lat_start), (lon_start, lat_stop),(lon_stop, lat_stop), (lon_stop, lat_start)]).buffer(0.01)
buffer_IN = {'latitude': slice(buffer_IN.bounds[3], buffer_IN.bounds[1], None),
            'longitude': slice(buffer_IN.bounds[0], buffer_IN.bounds[2], None)}

subset_LastTimeSinceDist_cube = LastTimeSinceDist_cube.sel(time = DataConfig['end_year']).sel(buffer_IN).LandsatDisturbanceTime
subset_LastTimeSinceDist_cube = subset_LastTimeSinceDist_cube.where(subset_LastTimeSinceDist_cube>=1)

subset_clim_cube = clim_cube.sel(buffer_IN)[[x for x in DataConfig['features'] if "WorlClim" in x]].astype('float16')
subset_clim_cube = interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_LastTimeSinceDist_cube).sel(IN)
subset_CRparams_cube = CRparams_cube.sel(buffer_IN).astype('float16')
subset_CRparams_cube = interpolate_worlClim(source_ds = subset_CRparams_cube, target_ds = subset_LastTimeSinceDist_cube).sel(IN)
subset_LastTimeSinceDist = subset_LastTimeSinceDist_cube.sel(IN)     
subset_canopyHeight_cube = canopyHeight_cube.sel(IN).sel(time = upscaling_config['output_writer_params']['dims']['time'])
subset_canopyHeight_cube = subset_canopyHeight_cube.rename({list(set(list(subset_canopyHeight_cube.variables.keys())) - set(subset_canopyHeight_cube.coords))[0] : [x for x in DataConfig['features']  if "canopy_height" in x][0]}).astype('float16')
subset_canopyHeight_cube = subset_canopyHeight_cube.where(subset_canopyHeight_cube >0)
subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_canopyHeight_cube.time.values}, axis=list(subset_canopyHeight_cube.dims).index('time'))
subset_agb_std_cube    = agb_std_cube.sel(IN).astype('float16').sel(time = upscaling_config['output_writer_params']['dims']['time'])
subset_agb_std_cube    = subset_agb_std_cube[DataConfig['agb_var_cube'] + '_std']
            
mask_intact_forest = ~np.zeros((subset_canopyHeight_cube.sizes['latitude'], subset_canopyHeight_cube.sizes['longitude']), dtype=bool)
for _, row in intact_tropical_forest.iterrows():
    polygon = row.geometry
    polygon_mask = geometry_mask([polygon], out_shape=mask_intact_forest.shape, 
                                 transform=subset_canopyHeight_cube.rio.transform())
    
    if False in polygon_mask:
        mask_intact_forest[polygon_mask==False] = False
mask_intact_forest = mask_intact_forest.reshape(-1)

ML_pred_age_end_members = []
ML_pred_age_start_members = []
for run_ in np.arange(upscaling_config['num_members']):

    subset_agb_cube        = agb_cube.sel(IN).sel(members=run_).astype('float16').sel(time = upscaling_config['output_writer_params']['dims']['time'])
    subset_agb_cube        = subset_agb_cube[DataConfig['agb_var_cube']].to_dataset(name= [x for x in DataConfig['features']  if "agb" in x][0])
    
    subset_features_cube   = xr.merge([subset_agb_cube, subset_clim_cube, subset_canopyHeight_cube])
    subset_features_cube   = subset_features_cube.where(subset_LastTimeSinceDist==50) 
                          
    with open(study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = "Classifier", id_ = run_), 'rb') as f:
        classifier_config = pickle.load(f)
    best_classifier = classifier_config['best_model']
    features_classifier = classifier_config['selected_features']
    norm_stats_classifier = classifier_config['norm_stats']
    
    with open(study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = 'Regressor', id_ = run_), 'rb') as f:
        regressor_config = pickle.load(f)
    best_regressor = regressor_config['best_model']
    features_regressor = regressor_config['selected_features']
    norm_stats_regressor = regressor_config['norm_stats']
    
    all_features = list(np.unique(features_classifier + features_regressor))
                       
    X_upscale = []
    for var_name in all_features:
        X_upscale.append(subset_features_cube[var_name])
        
    X_upscale_flattened = []

    for arr in X_upscale:
        data = arr.data.flatten()
        X_upscale_flattened.append(data)
        
    X_upscale_flattened = da.array(X_upscale_flattened).transpose().compute()
    
    ML_pred_class = np.zeros(X_upscale_flattened.shape[0]) * np.nan
    ML_pred_age = np.zeros(X_upscale_flattened.shape[0]) * np.nan
    
    mask = (np.all(np.isfinite(X_upscale_flattened), axis=1)) 

    if (X_upscale_flattened[mask].shape[0]>0):
        index_mapping_class = [all_features.index(feature) for feature in features_classifier]
        index_mapping_reg = [all_features.index(feature) for feature in features_regressor]
        
        if algorithm == "XGBoost":
            dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_class])
            pred_class = (best_classifier.predict(dpred) > 0.5).astype('int16')
            
        elif algorithm == "AutoML":
            dpred =  X_upscale_flattened[mask][:, index_mapping_class]
            dpred = pd.DataFrame(dpred, columns = features_classifier)                         
            pred_class = best_classifier.predict(dpred).values
        
        else:
            dpred =  X_upscale_flattened[mask][:, index_mapping_class]
            pred_class = best_classifier.predict(dpred)
            
        ML_pred_class[mask] = pred_class
        
        if algorithm == "XGBoost":
            dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_reg])
            pred_reg= best_regressor.predict(dpred)
            
        else:
            dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
            pred_reg= best_regressor.predict(dpred)
        
        pred_reg[pred_reg>=DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0] -1
        pred_reg[pred_reg<1] = 1
        ML_pred_age[mask] = np.round(pred_reg).astype("int16")
        ML_pred_age[ML_pred_class==1] = DataConfig['max_forest_age'][0]
        ML_pred_age   = ML_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1)
        ML_pred_age_end = ML_pred_age[:, :, 1, :].reshape(-1)
        ML_pred_age_end[~mask_intact_forest] = DataConfig['max_forest_age'][0] 
        ML_pred_age_start = ML_pred_age[:, :, 0, :].reshape(-1)
        
    else:
        # nothing to predict here -> just NaN arrays with correct shape
        ML_pred_age_start = np.full(
            len(subset_features_cube.latitude) * len(subset_features_cube.longitude),
            np.nan, dtype="float32"
        )
        ML_pred_age_end = np.full_like(ML_pred_age_start, np.nan)
    ML_pred_age_end_members.append(ML_pred_age_end)
    ML_pred_age_start_members.append(ML_pred_age_start)
        
ML_pred_age_end_members = np.stack(ML_pred_age_end_members, axis=0)
ML_pred_age_start_members = np.stack(ML_pred_age_start_members, axis=0)
sigma_ml_end = np.nanstd(ML_pred_age_end_members, axis=0, ddof=0)
sigma_ml_start = np.nanstd(ML_pred_age_start_members, axis=0, ddof=0)
sigma_B_meas_start = subset_agb_std_cube.sel(time = DataConfig['start_year']).values.reshape(-1)
sigma_B_meas_end = subset_agb_std_cube.sel(time = DataConfig['end_year']).values.reshape(-1)

for run_ in range(upscaling_config['num_members']):
    ML_pred_age_end   = ML_pred_age_end_members[run_, :]
    ML_pred_age_start = ML_pred_age_start_members[run_, :]
    subset_agb_cube   = agb_cube.sel(IN).sel(members=run_).astype('float16').sel(time = upscaling_config['output_writer_params']['dims']['time'])
    biomass_start     = subset_agb_cube.sel(time = DataConfig['start_year']).to_array().values.reshape(-1)
    biomass_end       = subset_agb_cube.sel(time = DataConfig['end_year']).to_array().values.reshape(-1)

    # --- CR-based bias correction ---
    cr_params = {
        "A": subset_CRparams_cube.A.values.reshape(-1),
        "b": subset_CRparams_cube.b.values.reshape(-1),
        "k": subset_CRparams_cube.k.values.reshape(-1)
    }
    cr_errors = {
        "A": subset_CRparams_cube.A_error.values.reshape(-1),
        "b": subset_CRparams_cube.b_error.values.reshape(-1),
        "k": subset_CRparams_cube.k_error.values.reshape(-1)
    }

    TSD = np.repeat(20, len(ML_pred_age_end))
    tmax = np.repeat(301, len(ML_pred_age_end))
    
    fusion = AgeFusion(config={
        "start_year": int(DataConfig['start_year'].split('-')[0]),
        "end_year": int(DataConfig['end_year'].split('-')[0])
    })

    corrected_pred_age_start, corrected_pred_age_end = fusion.fuse(
        ML_pred_age_start = ML_pred_age_start, ML_pred_age_end = ML_pred_age_end,
        LTSD = subset_LastTimeSinceDist.values.reshape(-1),
        biomass_start = biomass_start, biomass_end = biomass_end,
        cr_params = cr_params, cr_errors = cr_errors,
        ml_std_end = sigma_ml_end, ml_std_start = sigma_ml_start, 
        biomass_std_end = sigma_B_meas_end, biomass_std_start = sigma_B_meas_start, 
        TSD = TSD, tmax = tmax
    )
    
    # Reshape arrays
    fused_pred_age_start = corrected_pred_age_start.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
    fused_pred_age_end = corrected_pred_age_end.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
    
    # Create xarray dataset for each year
    ML_pred_age_start = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_start, 
                                                coords={"latitude": subset_features_cube.latitude, 
                                                        "longitude": subset_features_cube.longitude,
                                                        "time": [pd.to_datetime(DataConfig['start_year'])],                                                          
                                                        'members': [run_]}, 
                                                dims=["latitude", "longitude", "time", "members"])})
    
    ML_pred_age_end = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_end, 
                                                coords={"latitude": subset_features_cube.latitude, 
                                                        "longitude": subset_features_cube.longitude,
                                                        "time": [pd.to_datetime(DataConfig['end_year'])],                                                          
                                                        'members': [run_]}, 
                                                dims=["latitude", "longitude", "time", "members"])})
                  
    # Concatenate with the time dimensions and append the model member
    ds = xr.concat([ML_pred_age_start, ML_pred_age_end], dim= 'time').transpose('members', 'latitude', 'longitude', 'time')


#%% Plot bias correction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    ax_main.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
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
hat_t = ML_pred_age_start
sigma_ml = sigma_ml_start

A = cr_params['A']
b = cr_params['b']
k = cr_params['k']
m_fixed = 0.67

sd_A = cr_errors['A']
sd_b = cr_errors['b']
sd_k = cr_errors['k']

# Simulate observations
def cr_forward(t, A, b, k, m):
    p = 1.0 / (1.0 - m)
    u = np.clip(1.0 - b * np.exp(-k * t), 1e-12, 1.0 - 1e-12)
    return A * (u ** p)
    
B_true = cr_forward(hat_t, A, b, k, m_fixed)
B_obs = biomass_start * 0.47
sigma_B_meas = sigma_B_meas_start * 0.47

# Simulate corrections (in real use, call corrector.correct())
t_corrected = corrected_pred_age_start

# Single pixel detailed view
idx = 10000  # interesting pixel
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
    sigma_TSD=10.0
)
plt.savefig(f'/home/simon/Desktop/pixel_{idx}_diagnostic.png', dpi=300, bbox_inches='tight')    


