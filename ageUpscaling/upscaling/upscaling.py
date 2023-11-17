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
@Desc    :   A method class for upscaling MLP model
"""
import os
import shutil
from tqdm import tqdm
from itertools import product
from abc import ABC

import numpy as np
import yaml as yml
import pickle

import geopandas as gpd
from rasterio.features import geometry_mask

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
        buffer_IN = Polygon([(lon_start, lat_start), (lon_start, lat_stop),(lon_stop, lat_stop), (lon_stop, lat_start)]).buffer(0.01)
        buffer_IN = {'latitude': slice(buffer_IN.bounds[3], buffer_IN.bounds[1], None),
                     'longitude': slice(buffer_IN.bounds[0], buffer_IN.bounds[2], None)}
        
        
        subset_LastTimeSinceDist_cube = self.LastTimeSinceDist_cube.sel(IN).to_array()
        subset_LastTimeSinceDist_cube = subset_LastTimeSinceDist_cube.where(subset_LastTimeSinceDist_cube>=0).values.reshape(-1)
        
        if not np.isnan(subset_LastTimeSinceDist_cube).all():            
            
            subset_agb_cube        = self.agb_cube.sel(buffer_IN).astype('float16').sel(time = self.upscaling_config['output_writer_params']['dims']['time'])
            subset_agb_cube        = subset_agb_cube[self.DataConfig['agb_var_cube']].where(subset_agb_cube[self.DataConfig['agb_var_cube']] >0).to_dataset(name= [x for x in self.DataConfig['features']  if "agb" in x][0])
            
            subset_clim_cube = self.clim_cube.sel(buffer_IN)[[x for x in self.DataConfig['features'] if "WorlClim" in x]].astype('float16')
            subset_clim_cube = interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_agb_cube)
            subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_agb_cube.time.values}, axis=list(subset_agb_cube.dims).index('time'))
                        
            subset_canopyHeight_cube = self.canopyHeight_cube.sel(buffer_IN)
            subset_canopyHeight_cube = subset_canopyHeight_cube.rename({list(set(list(subset_canopyHeight_cube.variables.keys())) - set(subset_canopyHeight_cube.coords))[0] : [x for x in self.DataConfig['features']  if "canopy_height" in x][0]}).astype('float16')
            subset_canopyHeight_cube = subset_canopyHeight_cube.where(subset_canopyHeight_cube >0 )
            subset_canopyHeight_cube = subset_canopyHeight_cube.expand_dims({'time': subset_agb_cube.time.values}, axis=list(subset_agb_cube.dims).index('time'))
            
            subset_features_cube      = xr.merge([subset_agb_cube.sel(IN), subset_clim_cube.sel(IN), subset_canopyHeight_cube.sel(IN)]).transpose('latitude', 'longitude', 'time')
            
            output_reg_xr = []
            for run_ in np.arange(self.upscaling_config['num_members']):
                
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
                
                if (X_upscale_flattened[mask].shape[0]>0):
                    index_mapping_class = [all_features.index(feature) for feature in features_classifier]
                    index_mapping_reg = [all_features.index(feature) for feature in features_regressor]
                    
                    if self.algorithm == "XGBoost":
                        dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_class])
                        pred_class = (best_classifier.predict(dpred) > 0.5).astype('int16')
                    
                    elif self.algorithm == "MLP":
                        dpred =  X_upscale_flattened[mask][:, index_mapping_class]
                        dpred = np.stack([self.norm(dpred[:, features_classifier.index(var_name)], norm_stats_classifier[var_name]) for var_name in features_classifier], axis=1)
                    
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
                    pred_reg[pred_reg<1] = 1
                    ML_pred_age[mask] = np.round(pred_reg).astype("int16")
                    ML_pred_age[ML_pred_class==1] = self.DataConfig['max_forest_age'][0]
                    out_reg   = ML_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1)
                    
                    if self.upscaling_config['fuse_wLandsat']:
                        
                        subset_len = len(subset_LastTimeSinceDist_cube)
                        ML_pred_age_year2 = out_reg[:, :, 1, :].reshape(-1)
                                                
                        # Initialize with NaN values directly
                        fused_pred_age = np.full(subset_len, np.nan)
                        
                        # Stand replacement or afforestation occured and age ML is higher than Hansen Loss year
                        mask_Change1 = np.logical_and(subset_LastTimeSinceDist_cube <= 19, ML_pred_age_year2 > subset_LastTimeSinceDist_cube)
                        fused_pred_age[mask_Change1] = subset_LastTimeSinceDist_cube[mask_Change1]
                        
                        # Stand replacement or afforestation occured and age ML is lower or equal than Hansen Loss year
                        mask_Change2 = np.logical_and(subset_LastTimeSinceDist_cube <= 19, ML_pred_age_year2 <= subset_LastTimeSinceDist_cube)
                        fused_pred_age[mask_Change2] = ML_pred_age_year2[mask_Change2]
                                                
                        # Forest has been stable since 2000 or planted before 2000 and age ML is higher than 20
                        mask_intact1 = (subset_LastTimeSinceDist_cube >= 20)
                        fused_pred_age[mask_intact1] = ML_pred_age_year2[mask_intact1]
                                                        
                        fused_pred_age = fused_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1)                        
                        
                    output_data = {"forest_age_ML":xr.DataArray(out_reg, 
                                                                coords={"latitude": subset_features_cube.latitude, 
                                                                        "longitude": subset_features_cube.longitude,
                                                                        "time": subset_features_cube.time.values,                                                          
                                                                        'members': [run_]}, 
                                                                dims=["latitude", "longitude", "time", "members"])}
                    if self.upscaling_config['fuse_wLandsat']:
                        fused_pred_age = np.concatenate((out_reg[:, :, 0, :].reshape(out_reg.shape[0], out_reg.shape[1], 1, out_reg.shape[3]), fused_pred_age), axis=2)
                        output_data["forest_age_hybrid"] = xr.DataArray(fused_pred_age, 
                                                                       coords={"latitude": subset_features_cube.latitude, 
                                                                               "longitude": subset_features_cube.longitude,
                                                                               "time": subset_features_cube.time.values,                                                          
                                                                               'members': [run_]}, 
                                                                       dims=["latitude", "longitude", "time", "members"])
                    
                    ds = xr.Dataset(output_data)
                    
                    if self.upscaling_config['fuse_wLandsat']:
                        nan_mask = np.isnan(ds['forest_age_ML']) | np.isnan(ds['forest_age_hybrid'])
                        ds['forest_age_ML'] = ds['forest_age_ML'].where(~nan_mask, np.nan)
                        ds['forest_age_hybrid'] = ds['forest_age_hybrid'].where(~nan_mask, np.nan)
                    output_reg_xr.append(ds)
                            
            if len(output_reg_xr) >0:
                output_reg_xr = xr.concat(output_reg_xr, dim = 'members')
                mask = ~np.zeros(output_reg_xr['forest_age_ML'].isel(time=1, members=0).shape, dtype=bool)
                for _, row in self.intact_tropical_forest.iterrows():
                    polygon = row.geometry
                    polygon_mask = geometry_mask([polygon], out_shape=mask.shape, transform=output_reg_xr.rio.transform())
                    
                    if False in polygon_mask:
                        mask[polygon_mask==False] = False
                    
                mask= mask.reshape(output_reg_xr.latitude.shape[0], output_reg_xr.longitude.shape[0] , 1)
                out_2020 = output_reg_xr.sel(time='2020-01-01')
                out_2010 = output_reg_xr.sel(time='2020-01-01') - 10
                out_2010 = xr.where(out_2010 >= 0, out_2010, output_reg_xr.sel(time='2010-01-01'))
                out_2010 = out_2010.where(mask, self.DataConfig['max_forest_age'][0])
                out_2020 = out_2020.where(mask, self.DataConfig['max_forest_age'][0])                
                out_2010 = out_2010.where(out_2020<self.DataConfig['max_forest_age'][0], self.DataConfig['max_forest_age'][0])
                out_2010 = out_2010.where(np.isfinite(output_reg_xr.sel(time = '2020-01-01')))
                out_2020 = out_2020.where(np.isfinite(output_reg_xr.sel(time = '2020-01-01')))
                out_2010['time'] = xr.DataArray(np.array(["2010-01-01"], dtype="datetime64[ns]"), dims="time")                             
                output_reg_xr = xr.concat([out_2010, out_2020], dim= 'time').mean(dim= "members").transpose('latitude', 'longitude', 'time')                
                self.pred_cube.CubeWriter(output_reg_xr, n_workers=2)                
                # output_reg_xr_quantile = output_reg_xr.quantile([0.25, 0.75], dim="members")
                # output_reg_xr_iqr = output_reg_xr_quantile.sel(quantile = 0.75) - output_reg_xr_quantile.sel(quantile = 0.25)
                # output_reg_xr_median = output_reg_xr.median(dim= "members").transpose('latitude', 'longitude', 'time')                
                # output_reg_xr_median = output_reg_xr_median.rename({"forest_age_ML": "forest_age_ML_median", "forest_age_hybrid": "forest_age_hybrid_median"})
                # output_reg_xr_iqr = output_reg_xr_iqr.rename({"forest_age_ML": "forest_age_ML_IQR", "forest_age_hybrid": "forest_age_hybrid_IQR"})
                # self.pred_cube.CubeWriter(output_reg_xr_median, n_workers=2)
                # self.pred_cube.CubeWriter(output_reg_xr_iqr, n_workers=2)
                    
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
        
        for run_ in tqdm(np.arange(self.upscaling_config['num_members']), desc='Training model members'):
            
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
                                
        LatChunks = np.array_split(self.upscaling_config['output_writer_params']['dims']['latitude'], self.upscaling_config["num_chunks"])
        LonChunks = np.array_split(self.upscaling_config['output_writer_params']['dims']['longitude'], self.upscaling_config["num_chunks"])
        
        AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
        
        if  "SLURM_JOB_ID" in os.environ:
            selected_extent = AllExtents[task_id]
            
            self.process_chunk(selected_extent)
        
        else:
            if (self.n_jobs > 1):
            
                batch_size = 2
                for i in range(0, len(AllExtents), batch_size):
                    batch_futures = [self.process_chunk(extent) for extent in AllExtents[i:i+batch_size]]
                    dask.compute(*batch_futures, num_workers=self.n_jobs)
            
            else:
                for extent in tqdm(AllExtents, desc='Upscaling procedure'):
                    self.process_chunk(extent)   

        if os.path.exists(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/cube_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/cube_sync_{self.task_id}.zarrsync"))
        
    def process_chunk(self, extent):
        
        self._predict_func(extent).compute()
       
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
    
