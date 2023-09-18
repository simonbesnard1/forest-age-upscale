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
from ageUpscaling.methods.autoML import AutoML
from ageUpscaling.methods.feature_selection import FeatureSelection


algorithm = "XGBoost"
IN = {"latitude":slice(0, -1),
      "longitude":slice(-50, -49)}

study_dir = '/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_1km/XGBoost/version-1.0'

with open('/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/upscaling/100m/data_config_xgboost.yaml', 'r') as f:
    DataConfig =  yml.safe_load(f)

with open('/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/upscaling/100m/config_upscaling.yaml', 'r') as f:
    upscaling_config =  yml.safe_load(f)

lat_start, lat_stop = IN['latitude'].start, IN['latitude'].stop
lon_start, lon_stop = IN['longitude'].start, IN['longitude'].stop
buffer_IN = Polygon([(lon_start, lat_start), (lon_start, lat_stop),(lon_stop, lat_stop), (lon_stop, lat_start)]).buffer(0.01)
buffer_IN = {'latitude': slice(buffer_IN.bounds[3], buffer_IN.bounds[1], None),
             'longitude': slice(buffer_IN.bounds[0], buffer_IN.bounds[2], None)}

subset_LastTimeSinceDist_cube = xr.open_zarr(DataConfig['LastTimeSinceDist_cube']).sel(IN).isel(time =0).to_array()
subset_LastTimeSinceDist_cube = subset_LastTimeSinceDist_cube.where(subset_LastTimeSinceDist_cube>=0).values.reshape(-1)

if not np.isnan(subset_LastTimeSinceDist_cube).all():
    
    subset_agb_cube        = xr.open_zarr(DataConfig['agb_cube']).sel(buffer_IN).astype('float16').sel(time = upscaling_config['output_writer_params']['dims']['time'])
    subset_agb_cube        = subset_agb_cube[DataConfig['agb_var_cube']].where(subset_agb_cube[DataConfig['agb_var_cube']] >0).to_dataset(name= [x for x in DataConfig['features']  if "agb" in x][0])
    
    subset_clim_cube       = xr.open_zarr(DataConfig['clim_cube']).sel(buffer_IN)[[x for x in DataConfig['features'] if "WorlClim" in x]].astype('float16')
    subset_clim_cube =  interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_agb_cube)
    subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_agb_cube.time.values}, axis=list(subset_agb_cube.dims).index('time'))
    
    subset_canopyHeight_cube = xr.open_zarr(DataConfig['canopy_height_cube']).sel(buffer_IN).to_array().to_dataset(name= [x for x in DataConfig['features']  if "canopy_height" in x][0]).astype('float16')
    subset_canopyHeight_cube = subset_canopyHeight_cube.where(subset_canopyHeight_cube >0 )
    subset_canopyHeight_cube = subset_canopyHeight_cube.expand_dims({'time': subset_agb_cube.time.values}, axis=list(subset_agb_cube.dims).index('time'))
    
    subset_features_cube      = xr.merge([subset_agb_cube.sel(IN), subset_clim_cube.sel(IN), subset_canopyHeight_cube.sel(IN)])
    
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
        
        ML_pred_class = np.zeros(X_upscale_flattened.shape[0]) * np.nan
        ML_pred_age = np.zeros(X_upscale_flattened.shape[0]) * np.nan
        
        mask = (np.all(np.isfinite(X_upscale_flattened), axis=1)) 
        
        if (X_upscale_flattened[mask].shape[0]>0):
            index_mapping_class = [all_features.index(feature) for feature in features_classifier]
            index_mapping_reg = [all_features.index(feature) for feature in features_regressor]
            
            if algorithm == "XGBoost":
                dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_class])
                pred_class = (best_classifier.predict(dpred) > 0.5).astype(int)
            
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
            pred_reg[pred_reg<1] = 1
            ML_pred_age[mask] = np.round(pred_reg).astype("int16")
            ML_pred_age[ML_pred_class==1] = DataConfig['max_forest_age'][0]
            
            if upscaling_config['fuse_wLandsat']:
                fused_pred_age = np.empty(len(subset_LastTimeSinceDist_cube)) * np.nan
                
                # Stand replacement occured and age ML is higher than Hansen Loss year
                mask_lossYear1 = np.logical_and(subset_LastTimeSinceDist_cube <= 19, ML_pred_age > subset_LastTimeSinceDist_cube)
                fused_pred_age[mask_lossYear1] = subset_LastTimeSinceDist_cube[mask_lossYear1]
                
                # Stand replacement occured and age ML is lower or equal than Hansen Loss year
                mask_lossYear2 = np.logical_and(subset_LastTimeSinceDist_cube <= 19, ML_pred_age <= subset_LastTimeSinceDist_cube)
                fused_pred_age[mask_lossYear2] = ML_pred_age[mask_lossYear2]
                
                # Stand replacement occured and age ML is missing
                mask_lossYear3 = np.logical_and(subset_LastTimeSinceDist_cube <= 19, np.isnan(ML_pred_age) )
                fused_pred_age[mask_lossYear3] = subset_LastTimeSinceDist_cube[mask_lossYear3]
                
                # No stand replacement occured but gain and age ML is higher than 20
                mask_regrowth1 = np.logical_and(subset_LastTimeSinceDist_cube == 21, ML_pred_age > 19)
                fused_pred_age[mask_regrowth1] = 19
                
                # No stand replacement occured but gain and age ML is lower or equal than 20
                mask_regrowth2 = np.logical_and(subset_LastTimeSinceDist_cube == 21, ML_pred_age <= 19)
                fused_pred_age[mask_regrowth2] = ML_pred_age[mask_regrowth2]
                
                # No stand replacement occured but gain and age is missing
                mask_regrowth3 = np.logical_and(subset_LastTimeSinceDist_cube == 21, np.isnan(ML_pred_age))
                fused_pred_age[mask_regrowth3] = subset_LastTimeSinceDist_cube[mask_regrowth3]
                
                # Forest has been stable or growing the between 2000 and 2020 and age ML is higher or equal than 20
                # mask_intact1 = np.logical_and(subset_LastTimeSinceDist_cube == 300, ML_pred_age >= 20)
                # fused_pred_age[mask_intact1] = ML_pred_age[mask_intact1]
                
                # # Forest has been stable or growing the between 2000 and 2020 and age ML is lower than 20
                # mask_intact2 = np.logical_and(subset_LastTimeSinceDist_cube == 300, ML_pred_age < 20)
                # fused_pred_age[mask_intact2] = 20
                
                mask_intact1 = (subset_LastTimeSinceDist_cube == 300)
                fused_pred_age[mask_intact1] = ML_pred_age[mask_intact1]
                
                ML_pred_age[np.isnan(fused_pred_age)] = np.nan
                ML_pred_age[np.isnan(ML_pred_age)] = fused_pred_age[np.isnan(ML_pred_age)]
                
                fused_pred_age = fused_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1)                    
            
            out_reg   = ML_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1)
            output_data = {"forest_age_ML":xr.DataArray(out_reg, 
                                                        coords={"latitude": subset_features_cube.latitude, 
                                                                "longitude": subset_features_cube.longitude,
                                                                "time": subset_features_cube.time,                                                          
                                                                'members': [run_]}, 
                                                        dims=["latitude", "longitude", "time", "members"]).astype('float32')}
            if upscaling_config['fuse_wLandsat']:
                output_data["forest_age_hybrid"] = xr.DataArray(fused_pred_age, 
                                                               coords={"latitude": subset_features_cube.latitude, 
                                                                       "longitude": subset_features_cube.longitude,
                                                                       "time": subset_features_cube.time,                                                          
                                                                       'members': [run_]}, 
                                                               dims=["latitude", "longitude", "time", "members"]).astype('float32')
            
            output_reg_xr.append(xr.Dataset(output_data))