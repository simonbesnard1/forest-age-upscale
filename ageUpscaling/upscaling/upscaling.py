#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   upscaling.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for upscaling forest age ML model
"""
import os
import shutil
from tqdm import tqdm
from itertools import product
from abc import ABC
import subprocess
import glob

import numpy as np
import yaml as yml
import pickle

import geopandas as gpd
from rasterio.features import geometry_mask
import rioxarray as rio

import xarray as xr
import zarr
import dask
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
#from ageUpscaling.methods.autoML import AutoML
from ageUpscaling.methods.feature_selection import FeatureSelection

class UpscaleAge(ABC):
    """Study abstract class used for cross validation, model training, prediction.

    Parameters
    ----------
    DataConfig_path : DataConfig_path
        A data configuration path.     
    out_dir : str
        The study base directory.
        See `directory structure` for further details.
    exp_name : str = 'exp_name'
        The experiment name.
        See `directory structure` for further details.
    study_dir : Optional[str] = None
        The restore directory. If passed, an existing study is loaded.
        See `directory structure` for further details.
    n_jobs : int = 1
        Number of workers.

    """
    def __init__(self,
                 DataConfig_path: str,
                 upscaling_config_path: str,            
                 base_dir: str,
                 algorithm: str = 'MLP',
                 exp_name: str = None,
                 study_dir: str = None,
                 n_jobs: int = 1,
                 **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        
        with open(upscaling_config_path, 'r') as f:
            self.upscaling_config =  yml.safe_load(f)
            
        intact_forest = gpd.read_file(self.DataConfig['intact_forest_df'])
        self.intact_tropical_forest = intact_forest[intact_forest['IFL_ID'].str.contains('|'.join(['SAM', 'SEA', 'AFR']))]
        
        self.algorithm = algorithm
        self.base_dir = base_dir
        self.exp_name = exp_name
        
        if study_dir is None:
            study_dir = self.version_dir(self.base_dir, self.exp_name, self.algorithm)
            os.makedirs(study_dir, exist_ok=False)
        else:
            if not os.path.exists(study_dir):
                raise ValueError(f'restore path does not exist:\n{study_dir}')

        self.study_dir = study_dir
        self.n_jobs = n_jobs       
        self.valid_fraction= self.DataConfig["valid_fraction"]
        self.feature_selection= self.DataConfig["feature_selection"]
        self.feature_selection_method= self.DataConfig["feature_selection_method"]      
        self.upscaling_config['cube_location'] =  os.path.join(self.study_dir, self.upscaling_config['cube_name'])
        
        self.task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
        sync_file_features = os.path.abspath(f"{study_dir}/features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        
        self.agb_cube   = xr.open_zarr(self.DataConfig['agb_cube'], synchronizer=self.sync_feature)
        self.clim_cube  = xr.open_zarr(self.DataConfig['clim_cube'], synchronizer=self.sync_feature)
        self.canopyHeight_cube = xr.open_zarr(self.DataConfig['canopy_height_cube'], synchronizer=self.sync_feature)
        self.LastTimeSinceDist_cube = xr.open_zarr(self.DataConfig['LastTimeSinceDist_cube'], synchronizer=self.sync_feature)
                
        self.upscaling_config['sync_file_path'] = os.path.abspath(f"{study_dir}/cube_out_sync_{self.task_id}.zarrsync") 
        self.upscaling_config['output_writer_params']['dims']['latitude']  = self.agb_cube.latitude.values
        self.upscaling_config['output_writer_params']['dims']['longitude'] = self.agb_cube.longitude.values
        self.upscaling_config['output_writer_params']['dims']['members'] =  self.upscaling_config['num_members']
        
        self.pred_cube = DataCube(cube_config = self.upscaling_config)
        self.pred_cube.init_variable(self.upscaling_config['cube_variables'], 
                                     njobs= len(self.upscaling_config['cube_variables'].keys()))
        
    def version_dir(self, 
                    base_dir: str,
                    exp_name:str,
                    algorithm: str) -> str:
        """Creates a new version of a directory by appending the version number to the end of the directory name.
    
        If the directory already exists, it will be renamed to include the version number before the new directory is created.
        
        Parameters
        ----------
        base_dir : str
            The base directory where the new version of the study directory will be created.
        algorithm : str
            The name of the study directory.
            
        Returns
        -------
        str
            The full path to the new version of the study directory.
        """
        
        return self.increment_dir_version(base_dir,exp_name, algorithm)
    
    @staticmethod
    def increment_dir_version(base_dir: str,
                              exp_name:str,
                              algorithm:str) -> str:
        """Increments the version of a directory by appending the next available version number to the end of the directory name.
        
        Parameters
        ----------
        base_dir : str
            The base directory for the study.
        algorithm : str
            The name of the study.
        
        Returns
        -------
        str
            The name of the new directory with the incremented version number.
        """
        if not os.path.isdir(os.path.join(base_dir, exp_name, algorithm)):
            os.makedirs(os.path.join(base_dir, exp_name, algorithm))
        
        dir_list = [d for d in os.listdir(os.path.join(base_dir, exp_name, algorithm)) if d.startswith("version")]
        
        dir_list.sort()
        
        if len(dir_list) == 0:
            version = "1.0"
        else:
            last_dir = dir_list[-1]
            
            _, version = last_dir.split("-")
            
            major, minor = version.split(".")
            major = int(major)
            minor = int(minor)
            minor += 1
            if minor >= 10:
                major += 1
                minor = 0
            version = f"{major}.{minor}"
        
        return f"{base_dir}/{exp_name}/{algorithm}/version-{version}"
    
    @dask.delayed
    def _predict_func(self, 
                      IN) -> None:
          
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
        
        print(buffer_IN)
        subset_LastTimeSinceDist_cube = self.LastTimeSinceDist_cube.sel(time = self.DataConfig['end_year']).isel(buffer_IN)
        subset_LastTimeSinceDist_cube = subset_LastTimeSinceDist_cube.where(subset_LastTimeSinceDist_cube>=0)
        
        # if not np.isnan(subset_LastTimeSinceDist_cube.to_array().values).all():
                            
        subset_clim_cube = self.clim_cube.isel(buffer_IN)[[x for x in self.DataConfig['features'] if "WorlClim" in x]].astype('float16')
        subset_clim_cube = interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_LastTimeSinceDist_cube).isel(IN)
        subset_LastTimeSinceDist = subset_LastTimeSinceDist_cube.isel(IN).to_array().values.reshape(-1)      
        subset_canopyHeight_cube = self.canopyHeight_cube.isel(IN).sel(time = ['2000-01-01', '2020-01-01'])
        date_to_replace = pd.to_datetime('2000-01-01')
        new_date = pd.to_datetime('2010-01-01')
        time_index = subset_canopyHeight_cube.indexes['time']
        replace_index = time_index.get_loc(date_to_replace)
        subset_canopyHeight_cube['time'].values[replace_index] = new_date    
        subset_canopyHeight_cube = subset_canopyHeight_cube.rename({list(set(list(subset_canopyHeight_cube.variables.keys())) - set(subset_canopyHeight_cube.coords))[0] : [x for x in self.DataConfig['features']  if "canopy_height" in x][0]}).astype('float16')
        subset_canopyHeight_cube = subset_canopyHeight_cube.where(subset_canopyHeight_cube >0)
        subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_canopyHeight_cube.time.values}, axis=list(subset_canopyHeight_cube.dims).index('time'))
        mask_intact_forest = ~np.zeros(subset_LastTimeSinceDist_cube.LandsatDisturbanceTime.shape, dtype=bool)
        for _, row in self.intact_tropical_forest.iterrows():
            polygon = row.geometry
            polygon_mask = geometry_mask([polygon], out_shape=mask_intact_forest.shape, 
                                         transform=subset_LastTimeSinceDist_cube.rio.transform())
            
            if False in polygon_mask:
                mask_intact_forest[polygon_mask==False] = False
        mask_intact_forest = mask_intact_forest.reshape(-1)
        
        for run_ in np.arange(self.upscaling_config['num_members']):
            subset_agb_cube        = self.agb_cube.isel(IN).sel(members=run_).astype('float16').sel(time = self.upscaling_config['output_writer_params']['dims']['time'])
            #subset_agb_cube        = subset_agb_cube[self.DataConfig['agb_var_cube']].where(subset_agb_cube[self.DataConfig['agb_var_cube']] >0).to_dataset(name= [x for x in self.DataConfig['features']  if "agb" in x][0])
            subset_agb_cube        = subset_agb_cube[self.DataConfig['agb_var_cube']].to_dataset(name= [x for x in self.DataConfig['features']  if "agb" in x][0])
            
            subset_features_cube      = xr.merge([subset_agb_cube, subset_clim_cube, subset_canopyHeight_cube])
                                  
            with open(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = "Classifier", id_ = run_), 'rb') as f:
                classifier_config = pickle.load(f)
            best_classifier = classifier_config['best_model']
            features_classifier = classifier_config['selected_features']
            norm_stats_classifier = classifier_config['norm_stats']
            
            with open(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = 'Regressor', id_ = run_), 'rb') as f:
                regressor_config = pickle.load(f)
            best_regressor = regressor_config['best_model']
            features_regressor = regressor_config['selected_features']
            norm_stats_regressor = regressor_config['norm_stats']
            
            all_features = list(np.unique(features_classifier + features_regressor))
                               
            X_upscale = []
            for var_name in all_features:
                if self.algorithm == "MLP":
                    X_upscale.append(self.norm(subset_features_cube[var_name], norm_stats_classifier[var_name]))
                else:
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
                
                if self.algorithm == "XGBoost":
                    dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_class])
                    pred_class = (best_classifier.predict(dpred) > 0.5).astype('int16')
                
                elif self.algorithm == "MLP":
                    dpred =  X_upscale_flattened[mask][:, index_mapping_class]
                    dpred = np.stack([self.norm(dpred[:, features_classifier.index(var_name)], norm_stats_classifier[var_name]) for var_name in features_classifier], axis=1)
                    pred_class = best_classifier.predict(dpred)
                    
                elif self.algorithm == "AutoML":
                    dpred =  X_upscale_flattened[mask][:, index_mapping_class]
                    dpred = pd.DataFrame(dpred, columns = features_classifier)                         
                    pred_class = best_classifier.predict(dpred).values
                
                else:
                    dpred =  X_upscale_flattened[mask][:, index_mapping_class]
                    pred_class = best_classifier.predict(dpred)
                    
                ML_pred_class[mask] = pred_class
                
                if self.algorithm == "XGBoost":
                    dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_reg])
                    pred_reg= best_regressor.predict(dpred)
                    
                elif self.algorithm == "MLP":
                    dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
                    dpred = np.stack([self.norm(dpred[:, features_regressor.index(var_name)], norm_stats_regressor[var_name]) for var_name in features_regressor], axis=1)
                    pred_reg= best_regressor.predict(dpred)
                    
                elif self.algorithm == "AutoML":
                    dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
                    dpred = pd.DataFrame(dpred, columns = features_regressor)                         
                    pred_reg= best_regressor.predict(dpred).values
                    
                else:
                    dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
                    pred_reg= best_regressor.predict(dpred)
                
                pred_reg[pred_reg>=self.DataConfig['max_forest_age'][0]] = self.DataConfig['max_forest_age'][0] -1
                pred_reg[pred_reg<0] = 0
                ML_pred_age[mask] = np.round(pred_reg).astype("int16")
                ML_pred_age[ML_pred_class==1] = self.DataConfig['max_forest_age'][0]
                ML_pred_age   = ML_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1)
                ML_pred_age_end = ML_pred_age[:, :, 1, :].reshape(-1)
                ML_pred_age_end[~mask_intact_forest] = self.DataConfig['max_forest_age'][0] 
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
                fused_pred_age_start = fused_pred_age_end - (int(self.DataConfig['end_year'].split('-')[0]) -  int(self.DataConfig['start_year'].split('-')[0]))
                mask_Change1 = (fused_pred_age_start <0)
                fused_pred_age_start[mask_Change1] = ML_pred_age_start[mask_Change1]
                fused_pred_age_start[fused_pred_age_end == self.DataConfig['max_forest_age'][0]] = self.DataConfig['max_forest_age'][0]
                
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
                                                                    "time": [pd.to_datetime(self.DataConfig['start_year'])],                                                          
                                                                    'members': [run_]}, 
                                                            dims=["latitude", "longitude", "time", "members"])})
                
                ML_pred_age_end = xr.Dataset({"forest_age":xr.DataArray(fused_pred_age_end, 
                                                            coords={"latitude": subset_features_cube.latitude, 
                                                                    "longitude": subset_features_cube.longitude,
                                                                    "time": [pd.to_datetime(self.DataConfig['end_year'])],                                                          
                                                                    'members': [run_]}, 
                                                            dims=["latitude", "longitude", "time", "members"])})
                              
                # Concatenate with the time dimensions and append the model member
                ds = xr.concat([ML_pred_age_start, ML_pred_age_end], dim= 'time').transpose('latitude', 'longitude', 'time', 'members')
                
                self.pred_cube.CubeWriter(ds, n_workers=1)                
                
    def model_tuning(self,
                     run_: int=1,
                     task_:str='Regressor',
                     feature_selection:bool=True,
                     feature_selection_method:str = 'recursive',                     
                     train_subset:dict ={},
                     valid_subset:dict ={}) -> None:
        """Perform model tuning using cross-validation.

        Parameters
        ----------
        run_ : int, optional
            Number of model runs. Default is 1.
        method : str, optional
            The type of model to use for training. Default is 'MLPRegressor'.
        
        Returns
        -------
        None
            The function does not return any values, but it updates the `self.best_model` attribute
            with the best model found during the tuning process.
        """
        if feature_selection:
            self.DataConfig['features_selected'] = FeatureSelection(method=task_, 
                                                                   feature_selection_method = feature_selection_method, 
                                                                   features = self.DataConfig['features'],
                                                                   data = xr.open_dataset(self.DataConfig['training_dataset'])).get_features(n_jobs = self.n_jobs)
                
        else: 
            self.DataConfig['features_selected'] = self.DataConfig['features'].copy()
            
        if self.algorithm == "MLP":
            
            ml_method = MLPmethod(study_dir=self.study_dir, 
                                   DataConfig= self.DataConfig,
                                   method=self.algorithm + task_)
        elif self.algorithm == "XGBoost":
            ml_method = XGBoost(study_dir=self.study_dir, 
                                DataConfig= self.DataConfig,
                                method=self.algorithm + task_)
        elif self.algorithm == "RandomForest":
            ml_method = RandomForest(study_dir=self.study_dir, 
                                     DataConfig= self.DataConfig,
                                     method=self.algorithm + task_)
        # elif self.algorithm == "AutoML":
        #     ml_method = AutoML(study_dir=self.study_dir, 
        #                        DataConfig= self.DataConfig,
        #                        method=self.algorithm + task_)
        
        ml_method.train(train_subset=train_subset,
                        valid_subset=valid_subset,
                        n_jobs = self.n_jobs)
        
        if not os.path.exists(self.study_dir + '/save_model/'):
             os.makedirs(self.study_dir + '/save_model/')
             
        with open(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = task_, id_ = run_), "wb") as fout:
            
            pickle.dump({'best_model': ml_method.best_model, 
                         'selected_features': self.DataConfig['features_selected'], 
                         'norm_stats' : ml_method.mldata.norm_stats}, fout)
    
    def model_training(self) -> None:
        """Perform forward run of the model, which consists of generating high resolution maps of age using the trained model.

        Parameters
        ----------
        tree_cover_tresholds : dict[str, Any], optional
            Dictionary of tree cover tresholds to use for the forward run, default is {'000', '005', '010', '015', '020', '030'}
        nLatChunks : int, optional
            Number of chunks to use in the latitude dimension, default is 50
        nLonChunks : int, optional
            Number of chunks to use in the longitude dimension, default is 50
        high_res_pred : bool, optional
            Boolean indicating whether to perform high resolution prediction, default is False
        """
        
        cluster_ = xr.open_dataset(self.DataConfig['training_dataset']).cluster.values
        train_subset, valid_subset = train_test_split(cluster_, test_size=self.DataConfig['valid_fraction'], shuffle=True)
        
        for run_ in tqdm(np.arange(self.upscaling_config['n_members']), desc='Training model members'):
            
            for task_ in ["Regressor", "Classifier"]:
                if not os.path.exists(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = task_, id_ = run_)):
                    self.model_tuning(run_ = run_, 
                                      task_ = task_,
                                      feature_selection= self.DataConfig['feature_selection'],
                                      feature_selection_method = self.DataConfig['feature_selection_method'],     
                                      train_subset=train_subset, 
                                      valid_subset=valid_subset)
        
        if os.path.exists(os.path.join(self.study_dir, "tune")):
            shutil.rmtree(os.path.join(self.study_dir, "tune"))
            
        if os.path.exists(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/cube_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/cube_sync_{self.task_id}.zarrsync"))
        
    def ForwardRun(self, 
                   task_id=None) -> None:
                                
        lat_chunk_size, lon_chunk_size = self.pred_cube.cube.chunks['latitude'][0], self.pred_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.pred_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.pred_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
     
            # Select the extent based on the slice indices
            selected_extent = {"latitude": lat_slice, "longitude": lon_slice}
            selected_extent = {"longitude":slice(131062, 132187),
                               "latitude":slice(95625, 96187)}

            # Process the chunk
            self.process_chunk(selected_extent)
        
        else:
           print(f"Task ID {task_id} is out of range. No chunk to process.")

        if os.path.exists(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync"))
        
    def process_chunk(self, extent):
        
        self._predict_func(extent).compute()
        
    def AgeResample(self) -> None:
        """
            Calculate the age fraction.
            
            This function processes forest age class data using the Global Age Mapping Integration dataset,
            calculating the age fraction distribution changes over time. The results are saved as raster files.
            
            Attributes:
            - age_class_ds: The dataset containing age class information.
            - zarr_out_: An array to store the output data.
            
            The function performs the following operations:
            - Reads age class data.
            - Loops through each variable and age class in the dataset.
            - Transforms and attributes data.
            - Splits the geographical area into chunks.
            - Processes each chunk for each year.
            - Saves processed data as raster files.
            - Merges and converts the output into Zarr format.
        """
           
        
             
        age_cube = xr.open_zarr(self.upscaling_config['cube_location'])
        zarr_out_ = []
        
        for var_ in set(age_cube.variables.keys()) - set(age_cube.dims):
            
            LatChunks = np.array_split(age_cube.latitude.values, 3)
            LonChunks = np.array_split(age_cube.longitude.values, 3)
            chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
        		        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
        		    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] 
            
            iter_ = 0
            for chunck in chunk_dict:
                
                data_chunk = age_cube[var_].sel(chunck).transpose('time', 'latitude', 'longitude')
                data_chunk = data_chunk.where(np.isfinite(data_chunk), -9999).astype('float32')  
                
                data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
                data_chunk.attrs = {'long_name': 'Forest age',
                                    'units': 'Ton /ha',
                                    'valid_max': 300,
                                    'valid_min': 0}
                data_chunk.attrs["_FillValue"] = -9999  
                out_dir = '{study_dir}/tmp/{var_}/'.format(study_dir = self.study_dir, var_ = var_)
                if not os.path.exists(out_dir):
           		    os.makedirs(out_dir)
                       
                data_chunk.rio.to_raster(raster_path= out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)), 
                                         driver="COG", BIGTIFF='YES', compress=None,  dtype= 'float32')      
                
                gdalwarp_command = [
                                    'gdal_translate',
                                    '-a_nodata', '-9999',
                                    out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)),
                                    out_dir + '{var_}_{iter_}_nodata.tif'.format(var_ = var_, iter_=str(iter_))                
                                ]
                subprocess.run(gdalwarp_command, check=True)
                os.remove(out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)))
                
                iter_ += 1
        
            input_files = glob.glob(os.path.join(out_dir, '*_nodata.tif'))
            vrt_filename = out_dir + '/{var_}.vrt'.format(var_ = var_)
                
            gdalbuildvrt_command = [
                'gdalbuildvrt',
                vrt_filename
            ] + input_files
                
            subprocess.run(gdalbuildvrt_command, check=True)
                
            gdalwarp_command = [
                'gdalwarp',
                '-srcnodata', '-9999',
                '-dstnodata', '-9999',
                '-tr', str(self.upscaling_config['resample_resolution']), str(self.upscaling_config['resample_resolution']),
                '-t_srs', 'EPSG:4326',
                '-of', 'Gtiff',
                '-te', '-180', '-90', '180', '90',
                '-r', 'average',
                '-ot', 'Float32',
                '-co', 'COMPRESS=LZW',
                '-co', 'BIGTIFF=YES',
                '-overwrite',
                f'/{vrt_filename}',
               self.study_dir + f'/{var_}_{self.upscaling_config["resample_resolution"]}deg.tif'.format(var_=var_),
            ]
            subprocess.run(gdalwarp_command, check=True)
            
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
                
            da_ =  rio.open_rasterio(self.study_dir + f'/{var_}_{self.upscaling_config["resample_resolution"]}deg.tif'.format(var_=var_))
            da_ =  da_.rename({"x": 'longitude', "y": 'latitude', 'band': "time"}).to_dataset(name = var_)
            da_['time'] =  age_cube.time
            
            zarr_out_.append(da_)
        
        xr.merge(zarr_out_).to_zarr(self.study_dir + '/ForestAge_{resolution}deg'.format(resolution = str(self.upscaling_config['resample_resolution'])), mode= 'w')
        
        tif_files = glob.glob(os.path.join(self.study_dir, '*.tif'))
        for tif_file in tif_files:
              os.remove(tif_file)
       
    def norm(self, 
             x: np.array,
             norm_stats:dict) -> np.array:
        """Normalize an array of values using the given normalization statistics.

        Parameters
        ----------
        x : np.array
            The array of values to normalize.
        norm_stats : dict
            A dictionary containing the normalization statistics, with keys 'mean' and 'std'.
    
        Returns
        -------
        np.array
            The normalized array of values.
        """
        
        return (x - norm_stats['mean']) / norm_stats['std'] 
    
    def denorm_target(self, 
                      x: np.array,
                      norm_stats:dict) -> np.array:
        """De-normalize an array of values using the given normalization statistics.

        Parameters
        ----------
        x : np.array
            The array of values to de-normalize.
        norm_stats : dict
            A dictionary containing the normalization statistics, with keys 'mean' and 'std'.
    
        Returns
        -------
        np.array
            The de-normalized array of values.
        """
        
        return x * norm_stats['std'] + norm_stats['mean']
    
