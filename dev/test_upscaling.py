#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:08:42 2023

@author: simon
"""
import os
import shutil
from tqdm import tqdm
import atexit
from itertools import product
from abc import ABC

import numpy as np
import yaml as yml
import pickle

import multiprocessing
import xarray as xr
import zarr
import dask.array as da
from shapely.geometry import Polygon
import pandas as pd

from sklearn.model_selection import train_test_split
import xgboost as xgb

from ageUpscaling.core.cube import DataCube
from ageUpscaling.transformers.spatial import interpolate_worlClim
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.methods.xgboost import XGBoost
from ageUpscaling.methods.RandomForest import RandomForest
from ageUpscaling.methods.feature_selection import FeatureSelection

import geopandas as gpd
from rasterio.features import geometry_mask



LastTimeSinceDist_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/LandsatDisturbanceTime_100m')
agb_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/biomass_baccini_100m')
clim_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/WorlClim_1km')
canopyHeight_cube = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/canopyHeight_potapov_100m')

algorithm = "XGBoost"
IN = {'latitude': slice(8.99955555555556, 0.00044444444444025066, None),
      'longitude': slice(-71.99955555555556, -54.00044444444444, None)}
IN = {"latitude":slice(5, 4.5),
      "longitude":slice(-63.5, -62.5)}


study_dir = '/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.0'

with open('/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/upscaling/100m/data_config_xgboost.yaml', 'r') as f:
    DataConfig =  yml.safe_load(f)

with open('/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/upscaling/100m/config_upscaling.yaml', 'r') as f:
    upscaling_config =  yml.safe_load(f)

intact_forest = gpd.read_file(DataConfig['intact_forest_df'])
intact_tropical_forest = intact_forest[intact_forest['IFL_ID'].str.contains('|'.join(['SAM', 'SEA', 'AFR']))]

lat_start, lat_stop = IN['latitude'].start, IN['latitude'].stop
lon_start, lon_stop = IN['longitude'].start, IN['longitude'].stop
buffer_IN = Polygon([(lon_start, lat_start), (lon_start, lat_stop),(lon_stop, lat_stop), (lon_stop, lat_start)]).buffer(0.01)
buffer_IN = {'latitude': slice(buffer_IN.bounds[3], buffer_IN.bounds[1], None),
             'longitude': slice(buffer_IN.bounds[0], buffer_IN.bounds[2], None)}

subset_LastTimeSinceDist_cube = LastTimeSinceDist_cube.sel(time = DataConfig['end_year']).sel(IN).to_array()
subset_LastTimeSinceDist_cube = subset_LastTimeSinceDist_cube.where(subset_LastTimeSinceDist_cube>=0).values.reshape(-1) + 1

if not np.isnan(subset_LastTimeSinceDist_cube).all():            
    
    subset_agb_cube        = agb_cube.sel(time = DataConfig['start_year']).sel(buffer_IN).astype('float16')
    
    subset_agb_cube        = subset_agb_cube[DataConfig['agb_var_cube']].where(subset_agb_cube[DataConfig['agb_var_cube']] >0).to_dataset(name= [x for x in DataConfig['features']  if "agb" in x][0])
    
    subset_clim_cube = clim_cube.sel(buffer_IN)[[x for x in DataConfig['features'] if "WorlClim" in x]].astype('float16')
    subset_clim_cube = interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_agb_cube)
                
    subset_canopyHeight_cube = canopyHeight_cube.sel(time = DataConfig['start_year']).sel(buffer_IN)
    
    subset_canopyHeight_cube = subset_canopyHeight_cube.rename({list(set(list(subset_canopyHeight_cube.variables.keys())) - set(subset_canopyHeight_cube.coords))[0] : [x for x in DataConfig['features']  if "canopy_height" in x][0]}).astype('float16')
    subset_canopyHeight_cube = subset_canopyHeight_cube.where(subset_canopyHeight_cube >0 )
    
    subset_features_cube      = xr.merge([subset_agb_cube.sel(IN), subset_clim_cube.sel(IN), subset_canopyHeight_cube.sel(IN)])
    
    mask_intact_forest = ~np.zeros(subset_features_cube.canopy_height_gapfilled.shape, dtype=bool)
    for _, row in intact_tropical_forest.iterrows():
        polygon = row.geometry
        polygon_mask = geometry_mask([polygon], out_shape=mask_intact_forest.shape, transform=subset_features_cube.rio.transform())
        
        if False in polygon_mask:
            mask_intact_forest[polygon_mask==False] = False
    mask_intact_forest = mask_intact_forest.reshape(-1)
    
    output_reg_xr = []
    for run_ in np.arange(upscaling_config['num_members']):
        
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
        
        ML_pred_class_start = np.zeros(X_upscale_flattened.shape[0]) * np.nan
        ML_pred_age_start = np.zeros(X_upscale_flattened.shape[0]) * np.nan
        
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
                
            ML_pred_class_start[mask] = pred_class
            
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
            pred_reg[pred_reg<1] = 1
            ML_pred_age_start[mask] = np.round(pred_reg).astype("int16")
            ML_pred_age_start[ML_pred_class_start==1] = DataConfig['max_forest_age'][0]
            ML_pred_age_start[~mask_intact_forest] = DataConfig['max_forest_age'][0]  
            ML_pred_age_start[ML_pred_age_start>DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0]
            
            ML_pred_age_end = ML_pred_age_start + (int(DataConfig['end_year'].split('-')[0]) -  int(DataConfig['start_year'].split('-')[0]))
            ML_pred_age_end[ML_pred_age_end>DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0]
            
            # Initialize with NaN values directly
            fused_pred_age_end = np.full(len(ML_pred_age_end), np.nan)
            
            # Stand replacement occured and age ML is higher than Hansen Loss year
            mask_Change1 = np.logical_and(subset_LastTimeSinceDist_cube <= 20, ML_pred_age_end > subset_LastTimeSinceDist_cube)
            fused_pred_age_end[mask_Change1] = subset_LastTimeSinceDist_cube[mask_Change1]
            
            # Stand replacement occured and age ML is lower or equal than Hansen Loss year
            mask_Change2 = np.logical_and(subset_LastTimeSinceDist_cube <= 20, ML_pred_age_end <= subset_LastTimeSinceDist_cube)
            fused_pred_age_end[mask_Change2] = ML_pred_age_end[mask_Change2]
            
            # Afforestation occured and age ML is higher than Hansen Loss year
            mask_Change1 = np.logical_and(subset_LastTimeSinceDist_cube == 21, ML_pred_age_end > subset_LastTimeSinceDist_cube)
            fused_pred_age_end[mask_Change1] = 20
            
            # Afforestation occured and age ML is lower or equal than Hansen Loss year
            mask_Change2 = np.logical_and(subset_LastTimeSinceDist_cube == 21, ML_pred_age_end <= subset_LastTimeSinceDist_cube)
            fused_pred_age_end[mask_Change2] = ML_pred_age_end[mask_Change2]
            
            # Forest has been stable since 2000 or planted before 2000 and age ML is higher than 20
            mask_intact1 = (subset_LastTimeSinceDist_cube > 21)
            fused_pred_age_end[mask_intact1] = ML_pred_age_end[mask_intact1]
            
            fused_pred_age_mid = fused_pred_age_end - (int(DataConfig['end_year'].split('-')[0]) -  int(DataConfig['mid_year'].split('-')[0]))
            mask_Change1 = (fused_pred_age_mid <1)
            fused_pred_age_mid[mask_Change1] = ML_pred_age_start[mask_Change1] + (int(DataConfig['mid_year'].split('-')[0]) -  int(DataConfig['start_year'].split('-')[0]))
            fused_pred_age_mid[fused_pred_age_end == DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0]
            fused_pred_age_mid[fused_pred_age_mid>DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0]
            
            fused_pred_age_start = fused_pred_age_end - (int(DataConfig['end_year'].split('-')[0]) -  int(DataConfig['start_year'].split('-')[0]) -1)
            mask_Change1 = (fused_pred_age_start <1)
            fused_pred_age_start[mask_Change1] = ML_pred_age_start[mask_Change1] + 1
            fused_pred_age_start[fused_pred_age_end == DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0]
            fused_pred_age_start[fused_pred_age_start>DataConfig['max_forest_age'][0]] = DataConfig['max_forest_age'][0]
            
            # Mask nan consistenlty acroass years
            nan_mask = np.isnan(subset_LastTimeSinceDist_cube)
            fused_pred_age_end[nan_mask] = np.nan
            fused_pred_age_start[nan_mask] = np.nan
            fused_pred_age_mid[nan_mask] = np.nan
            
            # Reshape array
            fused_pred_age_start = fused_pred_age_start.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
            fused_pred_age_mid = fused_pred_age_mid.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
            fused_pred_age_end = fused_pred_age_end.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1) 
                        
            ML_pred_age_start = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_start, 
                                                        coords={"latitude": subset_features_cube.latitude, 
                                                                "longitude": subset_features_cube.longitude,
                                                                "time": [pd.to_datetime(DataConfig['start_year']) + pd.DateOffset(years=1)],                                                          
                                                                'members': [run_]}, 
                                                        dims=["latitude", "longitude", "time", "members"])})
            
            ML_pred_age_end = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_end, 
                                                        coords={"latitude": subset_features_cube.latitude, 
                                                                "longitude": subset_features_cube.longitude,
                                                                "time": [pd.to_datetime(DataConfig['end_year'])],                                                          
                                                                'members': [run_]}, 
                                                        dims=["latitude", "longitude", "time", "members"])})
            ML_pred_age_mid = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_mid, 
                                                        coords={"latitude": subset_features_cube.latitude, 
                                                                "longitude": subset_features_cube.longitude,
                                                                "time": [pd.to_datetime(DataConfig['mid_year'])],                                                          
                                                                'members': [run_]}, 
                                                        dims=["latitude", "longitude", "time", "members"])})                   
                                           
            ds = xr.concat([ML_pred_age_start, ML_pred_age_mid, ML_pred_age_end], dim= 'time')              
            
            output_reg_xr.append(ds)
                    
    if len(output_reg_xr) >0:
        output_reg_xr = xr.concat(output_reg_xr, dim = 'members').mean(dim= "members").transpose('latitude', 'longitude', 'time')        
