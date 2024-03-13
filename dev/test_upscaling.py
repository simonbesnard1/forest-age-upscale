#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:08:42 2023

@author: simon
"""

import numpy as np
import yaml as yml
import pickle

import xarray as xr
import dask.array as da
import pandas as pd

import xgboost as xgb

from ageUpscaling.transformers.spatial import interpolate_worlClim
import geopandas as gpd
from rasterio.features import geometry_mask

LastTimeSinceDist_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/LandsatDisturbanceTime_100m')
agb_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v4_members')
clim_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/WorlClim_1km')
canopyHeight_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/canopyHeight_potapov_100m')

algorithm = "XGBoost"
IN = {'latitude': slice(8.99955555555556, 0.00044444444444025066, None),
      'longitude': slice(-71.99955555555556, -54.00044444444444, None)}
IN = {"latitude":slice(5, 4.5),
      "longitude":slice(-63.5, -62.5)}
IN = {"longitude":slice(131062, 132187),
      "latitude":slice(95625, 96187)}

run_ =10 

study_dir = '/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.2'

with open('/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/upscaling/100m/data_config_xgboost.yaml', 'r') as f:
    DataConfig =  yml.safe_load(f)

with open('/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/upscaling/100m/config_upscaling.yaml', 'r') as f:
    upscaling_config =  yml.safe_load(f)

intact_forest = gpd.read_file(DataConfig['intact_forest_df'])
intact_tropical_forest = intact_forest[intact_forest['IFL_ID'].str.contains('|'.join(['SAM', 'SEA', 'AFR']))]

lat_start, lat_stop = IN['latitude'].start, IN['latitude'].stop
lon_start, lon_stop = IN['longitude'].start, IN['longitude'].stop
#buffer_IN = Polygon([(lon_start, lat_start), (lon_start, lat_stop),(lon_stop, lat_stop), (lon_stop, lat_start)]).buffer(20)
#buffer_IN = {'latitude': slice(buffer_IN.bounds[3], buffer_IN.bounds[1], None),
#             'longitude': slice(buffer_IN.bounds[0], buffer_IN.bounds[2], None)}

if (lat_start == 0) and (lon_start == 0) :
    buffer_IN = {'latitude': slice(0, lat_stop+20, None),
                 'longitude': slice(0, lon_stop+20, None)}
elif (lat_start == 0) and (lon_start > 0):   
    buffer_IN = {'latitude': slice(0, lat_stop+20, None),
                 'longitude': slice(lon_start+20, lon_stop+20, None)}
elif (lat_start > 0) and (lon_start == 0):   
    buffer_IN = {'latitude': slice(lat_start+20, lat_stop+20, None),
                 'longitude': slice(0, lon_stop+20, None)}
else:
    buffer_IN = {'latitude': slice(lat_start+20, lat_stop+20, None),
                 'longitude': slice(lon_start+20, lon_stop+20, None)}


subset_LastTimeSinceDist_cube = LastTimeSinceDist_cube.sel(time = DataConfig['end_year']).isel(buffer_IN)
subset_LastTimeSinceDist_cube = subset_LastTimeSinceDist_cube.where(subset_LastTimeSinceDist_cube>=0)

# if not np.isnan(subset_LastTimeSinceDist_cube.to_array().values).all():
                    
subset_clim_cube = clim_cube.isel(buffer_IN)[[x for x in DataConfig['features'] if "WorlClim" in x]].astype('float16')
subset_clim_cube = interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_LastTimeSinceDist_cube).isel(IN)
subset_LastTimeSinceDist = subset_LastTimeSinceDist_cube.isel(IN).to_array().values.reshape(-1)      
subset_canopyHeight_cube = canopyHeight_cube.isel(IN).sel(time = ['2000-01-01', '2020-01-01'])
date_to_replace = pd.to_datetime('2000-01-01')
new_date = pd.to_datetime('2010-01-01')
time_index = subset_canopyHeight_cube.indexes['time']
replace_index = time_index.get_loc(date_to_replace)
subset_canopyHeight_cube['time'].values[replace_index] = new_date    
subset_canopyHeight_cube = subset_canopyHeight_cube.rename({list(set(list(subset_canopyHeight_cube.variables.keys())) - set(subset_canopyHeight_cube.coords))[0] : [x for x in DataConfig['features']  if "canopy_height" in x][0]}).astype('float16')
subset_canopyHeight_cube = subset_canopyHeight_cube.where(subset_canopyHeight_cube >0)
subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_canopyHeight_cube.time.values}, axis=list(subset_canopyHeight_cube.dims).index('time'))
            
mask_intact_forest = ~np.zeros(subset_LastTimeSinceDist_cube.LandsatDisturbanceTime.shape, dtype=bool)
for _, row in intact_tropical_forest.iterrows():
    polygon = row.geometry
    polygon_mask = geometry_mask([polygon], out_shape=mask_intact_forest.shape, 
                                 transform=subset_LastTimeSinceDist_cube.rio.transform())
    
    if False in polygon_mask:
        mask_intact_forest[polygon_mask==False] = False
mask_intact_forest = mask_intact_forest.reshape(-1)

for run_ in np.arange(upscaling_config['num_members']):

    subset_agb_cube        = agb_cube.isel(IN).sel(members=run_).astype('float16').sel(time = upscaling_config['output_writer_params']['dims']['time'])
    subset_agb_cube        = subset_agb_cube[DataConfig['agb_var_cube']].to_dataset(name= [x for x in DataConfig['features']  if "agb" in x][0])
    
    subset_features_cube      = xr.merge([subset_agb_cube, subset_clim_cube, subset_canopyHeight_cube])
                          
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
    print((X_upscale_flattened[mask].shape[0]>0))
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
                
        elif algorithm == "AutoML":
            dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
            dpred = pd.DataFrame(dpred, columns = features_regressor)                         
            pred_reg= best_regressor.predict(dpred).values
            
        else:
            dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
            pred_reg= best_regressor.predict(dpred)
        
        pred_reg[pred_reg>=DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0] -1
        pred_reg[pred_reg<0] = 0
        ML_pred_age[mask] = np.round(pred_reg).astype("int16")
        ML_pred_age[ML_pred_class==1] = DataConfig['max_forest_age'][0]
        ML_pred_age   = ML_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1)
        ML_pred_age_end = ML_pred_age[:, :, 1, :].reshape(-1)
        ML_pred_age_end[~mask_intact_forest] = DataConfig['max_forest_age'][0] 
        ML_pred_age_start = ML_pred_age[:, :, 0, :].reshape(-1)
        
        # Initialize with NaN values directly
        fused_pred_age_end = np.full(len(ML_pred_age_end), np.nan)
        
        # Stand replacement occured and age ML is higher than Hansen Loss year
        mask_Change1 = np.logical_and(subset_LastTimeSinceDist <= 19, ML_pred_age_end > subset_LastTimeSinceDist)
        fused_pred_age_end[mask_Change1] = subset_LastTimeSinceDist[mask_Change1]
        
        # Stand replacement occured and age ML is lower or equal than Hansen Loss year
        mask_Change2 = np.logical_and(subset_LastTimeSinceDist <= 19, ML_pred_age_end <= subset_LastTimeSinceDist)
        fused_pred_age_end[mask_Change2] = ML_pred_age_end[mask_Change2]
        
        # Afforestation occured and age ML is higher than Hansen Loss year
        mask_Change1 = np.logical_and(subset_LastTimeSinceDist == 20, ML_pred_age_end > subset_LastTimeSinceDist)
        fused_pred_age_end[mask_Change1] = 20
        
        # Afforestation occured and age ML is lower or equal than Hansen Loss year
        mask_Change3 = np.logical_and(subset_LastTimeSinceDist == 20, ML_pred_age_end <= subset_LastTimeSinceDist)
        fused_pred_age_end[mask_Change3] = ML_pred_age_end[mask_Change3]
        
        # Forest has been stable since 2000 or planted before 2000 and age ML is higher than 20
        mask_intact1 = (subset_LastTimeSinceDist > 20)
        fused_pred_age_end[mask_intact1] = ML_pred_age_end[mask_intact1]
        
        # Backward fusion for the start year
        fused_pred_age_start = fused_pred_age_end - (int(DataConfig['end_year'].split('-')[0]) -  int(DataConfig['start_year'].split('-')[0]))
        mask_Change1 = (fused_pred_age_start <0)
        fused_pred_age_start[mask_Change1] = ML_pred_age_start[mask_Change1]
        fused_pred_age_start[fused_pred_age_end == DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0]
        
        # Mask nan consistenlty across years
        nan_mask = np.isnan(subset_LastTimeSinceDist)
        fused_pred_age_end[nan_mask] = np.nan
        fused_pred_age_start[nan_mask] = np.nan
        
        # Reshape arrays
        fused_pred_age_start = fused_pred_age_start.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
        fused_pred_age_end = fused_pred_age_end.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
        
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
        ds = xr.concat([ML_pred_age_start, ML_pred_age_end], dim= 'time').transpose('latitude', 'longitude', 'time', 'members')
                  
                                               