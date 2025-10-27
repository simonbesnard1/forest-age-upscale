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
#IN = {'latitude': slice(75.77733333333333, 72.22266666666667, None), 'longitude': slice(83.11155555555558, 86.66622222222225, None)}
IN = {'longitude': slice(10.416508, 10.487892 , None) , 'latitude': slice(51.101666, 51.056723 , None) }

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
        "end_year": int(DataConfig['end_year'].split('-')[0]),
        "sigma_TSD": 5.0
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
    
    # # Reshape arrays
    # fused_pred_age_start = corrected_pred_age_start.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
    # fused_pred_age_end = corrected_pred_age_end.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
    
    # # Create xarray dataset for each year
    # ML_pred_age_start = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_start, 
    #                                             coords={"latitude": subset_features_cube.latitude, 
    #                                                     "longitude": subset_features_cube.longitude,
    #                                                     "time": [pd.to_datetime(DataConfig['start_year'])],                                                          
    #                                                     'members': [run_]}, 
    #                                             dims=["latitude", "longitude", "time", "members"])})
    
    # ML_pred_age_end = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_end, 
    #                                             coords={"latitude": subset_features_cube.latitude, 
    #                                                     "longitude": subset_features_cube.longitude,
    #                                                     "time": [pd.to_datetime(DataConfig['end_year'])],                                                          
    #                                                     'members': [run_]}, 
    #                                             dims=["latitude", "longitude", "time", "members"])})
                  
    # # Concatenate with the time dimensions and append the model member
    # ds = xr.concat([ML_pred_age_start, ML_pred_age_end], dim= 'time').transpose('members', 'latitude', 'longitude', 'time')


#%% Plot bias correction
import numpy as np
import matplotlib.pyplot as plt

def cr_forward(t, A, b, k, m=1/3, eps=1e-9):
    """B_CR(t) = A * (1 - b * exp(-k t))^(1/(1-m))."""
    p = 1.0 / (1.0 - m)
    u = 1.0 - b * np.exp(-k * t)
    u = np.clip(u, eps, 1.0 - eps)
    return A * (u ** p)

def cr_sensitivities(t, A, b, k, m=1/3, eps=1e-9):
    """
    Return [dB/dA, dB/db, dB/dk] for fixed m (vectorized over t).
    Used for the delta-method uncertainty on B(t).
    """
    p = 1.0 / (1.0 - m)
    u = 1.0 - b * np.exp(-k * t)
    u = np.clip(u, eps, 1.0 - eps)
    u_pow_p      = u ** p
    u_pow_p_m1   = u ** (p - 1.0)
    dB_dA = u_pow_p
    dB_db = -A * p * u_pow_p_m1 * np.exp(-k * t)
    dB_dk =  A * p * u_pow_p_m1 * (b * t * np.exp(-k * t))
    return np.vstack([dB_dA, dB_db, dB_dk])  # shape (3, len(t))

def cr_sigma_B_delta(t, A, b, k, sd_A, sd_b, sd_k, m=1/3):
    """
    Delta-method var(B(t)) ≈ g^T Σ g with diagonal Σ = diag(sd_A^2, sd_b^2, sd_k^2).
    Returns 1σ at each t.
    """
    G = cr_sensitivities(t, A, b, k, m)                 # (3, T)
    s2 = (G[0]**2) * (sd_A**2) + (G[1]**2) * (sd_b**2) + (G[2]**2) * (sd_k**2)
    return np.sqrt(np.maximum(s2, 0.0))

def plot_age_pixel_diagnostic(
    A, b, k, sd_A, sd_b, sd_k,
    t_prior, t_post, B_obs, sigma_B,
    m=1/3,
    tmin=0.0, tmax=None,
    lat=None, lon=None,
    units="MgC ha$^{-1}$",
    ax=None
):
    """
    Pretty diagnostic for one pixel.

    Parameters
    ----------
    A,b,k, sd_* : floats
        CR parameters and their std errors (diagonal) for this pixel.
    t_prior, t_post : floats
        ML prior age and posterior/MAP age (years).
    B_obs, sigma_B : floats
        Observed biomass (same units as A) and its 1σ.
    m : float
        Fixed CR shape parameter (default 1/3).
    tmin, tmax : floats
        Plot range in years (tmax defaults to 1.2*max(t_prior, t_post) or 120 if both small).
    lat, lon : floats
        Optional location for the title.
    units : str
        Y-axis label units.
    ax : matplotlib axis
        If None, creates a new figure.
    """
    # choose sensible t-range
    mxt = np.nanmax([t_prior if np.isfinite(t_prior) else 0.0,
                     t_post  if np.isfinite(t_post)  else 0.0, 50.0])
    if tmax is None:
        tmax = max(120.0, 1.2 * mxt)
    T = np.linspace(max(0.0, tmin), tmax, 400)

    # CR curve and its 1σ ribbon from parameter uncertainty
    B_curve = cr_forward(T, A, b, k, m)
    B_sig   = cr_sigma_B_delta(T, A, b, k, sd_A, sd_b, sd_k, m)

    # implied biomasses at prior/post ages
    B_prior = cr_forward(t_prior, A, b, k, m) if np.isfinite(t_prior) else np.nan
    B_post  = cr_forward(t_post,  A, b, k, m) if np.isfinite(t_post)  else np.nan

    # plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))

    # CR curve ±1σ ribbon
    ax.plot(T, B_curve, lw=2.2, label="Chapman–Richards B(t)")
    ax.fill_between(T, B_curve - B_sig, B_curve + B_sig, alpha=0.18, linewidth=0, label="CR param ±1σ")

    # prior & posterior verticals
    if np.isfinite(t_prior):
        ax.axvline(t_prior, ls="--", lw=1.8, color="tab:blue", alpha=0.75, label=f"ML age (t) = {t_prior:.1f}")
        ax.scatter([t_prior], [B_prior], s=60, marker="s", color="tab:green", zorder=3,
                   label=f"B(t) = {B_prior:.1f}")
    if np.isfinite(t_post):
        ax.axvline(t_post, ls="--", lw=2.2, color="tab:blue", alpha=0.9, dashes=(3,2),
                   label=f"Posterior age (MAP) = {t_post:.1f}")
        ax.scatter([t_post], [B_post], s=70, marker="D", color="tab:red", zorder=3,
                   label=f"B(MAP) = {B_post:.1f}")

    # observed biomass with error bar
    if np.isfinite(B_obs) and np.isfinite(sigma_B):
        ax.errorbar([0.02*(tmax - tmin) + tmin], [B_obs], yerr=[sigma_B], fmt='o', ms=6,
                    color="tab:orange", ecolor="tab:orange", elinewidth=2, capsize=4,
                    label="Observed biomass ±σ")

    # cosmetics
    loc_txt = f" ({lat:.1f}N, {lon:.1f}E)" if (lat is not None and lon is not None) else ""
    ax.set_title(f"Pixel{loc_txt}", pad=10)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(f"Aboveground biomass (same units as Bmax/B_obs)\n[{units}]")
    ax.set_xlim(tmin, tmax)
    ymin = 0.0
    ymax = max(np.nanmax(B_curve + B_sig) * 1.05, (B_obs + sigma_B)*1.1 if np.isfinite(B_obs) else 0.0)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.25)

    return ax

# one pixel's values (units already matched: e.g., both in Mg C ha-1)
ax = plot_age_pixel_diagnostic(
    A=cr_params['A'][10], b=cr_params['b'][1], k=cr_params['k'][10],
    sd_A=cr_errors['A'][10], sd_b=cr_errors['b'][1], sd_k=cr_errors['k'][10],
    t_prior=ML_pred_age_start[10], t_post=corrected_pred_age_start[10],
    B_obs=biomass_start[10], sigma_B=sigma_B_meas_start[10],
    m=0.67,
    lat=45.1, lon=2.3,
    units="Mg C ha$^{-1}$"
)
plt.show()


