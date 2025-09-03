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
from ageUpscaling.methods.CRBayesAgeFuser import CRBayesAgeFuser
from ageUpscaling.methods.AgeFusion import AgeFusion

study_dir = '/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-3.0'

with open('/home/simon/hpc_home/projects/forest-age-upscale/config_files/upscaling/100m/data_config_xgboost.yaml', 'r') as f:
    DataConfig =  yml.safe_load(f)

with open('/home/simon/hpc_home/projects/forest-age-upscale/config_files/upscaling/100m/config_upscaling.yaml', 'r') as f:
    upscaling_config =  yml.safe_load(f)

LastTimeSinceDist_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/LandsatDisturbanceTime_100m')
agb_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v5_members')
clim_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/WorlClim_1km')
canopyHeight_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/canopyHeight_potapov_100m')
CRparams_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/CR_growth_curve_params_1km')
agb_std_cube = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v5')

intact_forest = gpd.read_file('/home/simon/hpc_home/projects/forest-age-upscale/data/shapefiles/IFL_2020_v2.zip')
intact_tropical_forest = intact_forest[intact_forest['IFL_ID'].str.contains('|'.join(['SAM', 'SEA', 'AFR']))]

algorithm = "XGBoost"
# bounding box for France
gdf = gpd.read_file("/home/simon/Downloads/boreal_tiles_v004_model_ready.gpkg")
gdf_wgs84 = gdf.to_crs(epsg=4326)
gdf_selected = gdf_wgs84[gdf_wgs84["tile_num"] == 37522]
lat_min, lat_max = gdf_selected.bounds['miny'].values[0], gdf_selected.bounds['maxy'].values[0]
lon_min, lon_max = gdf_selected.bounds['minx'].values[0], gdf_selected.bounds['maxx'].values[0]
IN = {'latitude': slice(lat_max, lat_min, None),
      'longitude': slice(lon_min, lon_max, None)}

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
        ML_pred_age_end_members.append(ML_pred_age_end)
        ML_pred_age_start_members.append(ML_pred_age_start)
        
ML_pred_age_end_members = np.stack(ML_pred_age_end_members, axis=0)
ML_pred_age_start_members = np.stack(ML_pred_age_start_members, axis=0)
sigma_ml_end = np.nanstd(ML_pred_age_end_members, axis=0)
sigma_ml_start = np.nanstd(ML_pred_age_start_members, axis=0)
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
        