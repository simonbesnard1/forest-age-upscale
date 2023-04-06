#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   extrapolation_index.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for calculating extrapolation index
"""
import os
import shutil
from abc import ABC
from itertools import product
import atexit

import numpy as np
import yaml as yml

import dask
import dask.array as da
import xarray as xr
import zarr

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import shap
from sklearn.model_selection import GridSearchCV,  KFold
from sklearn import preprocessing

from ageUpscaling.core.cube import DataCube
from ageUpscaling.methods.xgboost import XGBoost


synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class ExtrapolationIndex(ABC):
    """ExtrapolationIndex abstract class used for calculating extrapolation index

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
                 DataConfig_path: str=None,
                 cube_config_path: str=None,            
                 base_dir: str=None,
                 n_jobs: int = 1,
                 **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
            
        self.base_dir = base_dir
        self.n_jobs = n_jobs
        self.cube_config['cube_location'] = os.path.join(self.base_dir, self.cube_config['cube_name'])
        
    def trainKNN(self,
                 x_train, 
                 y_train, 
                 metric='euclidean', 
                 k_range = range(1, 31)):
        
        X = x_train.to_array().transpose('cluster','sample', 'variable').values.reshape(-1, len(self.DataConfig['features']))
        Y = y_train.to_array().values.reshape(-1)
        Y[Y<1] = 1 ## set min age to 1
        mask_nan = (np.all(np.isfinite(X), axis=1)) & (np.isfinite(Y))
        X, Y = X[mask_nan, :], Y[mask_nan]    
        param_grid = dict(n_neighbors=k_range)
        knn = KNeighborsRegressor(metric=metric)
        kf = KFold(n_splits=10)
        grid = GridSearchCV(knn, param_grid, cv=kf, scoring='neg_mean_squared_error')
        scaler = preprocessing.StandardScaler().fit(X)

        grid.fit(scaler.transform(X), Y)
        
        return grid, scaler
    
    def calculate_distance(self,
                           best_KNN,
                           scaler,
                           x_test):
        
        # distance_sum = 0
       
        # for i in range(grid.best_params_["n_neighbors"]):
        #     distance_sum += np.sum(list(weights[0].values()) * np.abs(x_test - x_train[indices[0][i]]), axis=1)
        # average_distance = distance_sum / grid.best_params_["n_neighbors"]
        
        distances, indices = best_KNN.best_estimator_.kneighbors(scaler.transform(x_test))

        # calculate the average distance for each new data point
        average_distance = np.mean(distances, axis=1)
       
        return average_distance

       
    def calculate_epsilon(self,
                          x_train, 
                          y_pred, 
                          y_obs):
        
        ##TODO currently not properly calculated
        delta = np.abs(y_pred - y_obs)
        
        # calculate the mean error of f
        mean_error = np.mean(delta)
        
        delta_diff = np.diff(delta)
        distance_diff = np.diff(x_train[:, -1])
        mean_delta_change = np.mean(delta_diff / distance_diff)
        
        # normalize the mean change of error by the mean error of f
        epsilon = mean_delta_change / mean_error
        
        return epsilon
        
    def calculate_weights(self,
                          X, Y):
        
        cluster_ = np.load(self.DataConfig['xval_index_path'])        
        train_subset, valid_subset = train_test_split(cluster_, test_size=self.DataConfig['valid_fraction'], shuffle=True)
        self.DataConfig['features_selected'] = self.DataConfig['features'] 
        ml_method = XGBoost(tune_dir=os.path.join(self.base_dir, "tune"), 
                            DataConfig= self.DataConfig,
                            method="XGBoostRegressor")
    
        ml_method.train(train_subset=train_subset,
                        valid_subset=valid_subset,
                        n_jobs = self.n_jobs)
        
        explainer = shap.TreeExplainer(ml_method.best_model)
        X = ml_method.mldata.train_dataloader().get_xy()['features']
        shap_values = explainer.shap_values(X)
        
        weights = [{self.DataConfig['features'][i]: np.median((np.abs(shap_values[:, i]) - np.min(np.abs(shap_values[:, i]))) / (np.max(np.abs(shap_values[:, i])) - np.min(np.abs(shap_values[:, i])))) for i in  np.arange(len(self.DataConfig['features']))}]
        
        shutil.rmtree(os.path.join(self.base_dir, "tune"))  
        
        return weights
   
    def calculate_index(self, 
                        IN,
                        best_KNN, 
                        scaler) -> None:
        
        subset_agb_cube  = xr.open_zarr(self.DataConfig['agb_cube'], synchronizer=synchronizer).sel(latitude= IN['latitude'],longitude=IN['longitude'])
        subset_clim_cube = xr.open_zarr(self.DataConfig['clim_cube'], synchronizer=synchronizer).sel(latitude= IN['latitude'],longitude=IN['longitude'])[[x for x in self.DataConfig['features'] if "WorlClim" in x]]
        
        if not self.cube_config["high_res_pred"]:
            subset_agb_cube    = subset_agb_cube.rename({'agb_001deg_cc_min_{tree_cover}'.format(tree_cover = self.tree_cover) : 'agb'})
       
        subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_agb_cube.time.values}, axis=list(subset_agb_cube.dims).index('time'))
        subset_agb_cube  = subset_agb_cube.agb.where(subset_agb_cube.agb >0).to_dataset()
        subset_cube      = xr.merge([subset_agb_cube, subset_clim_cube])
        
        X_upscale_reg = []
        for var_name in self.DataConfig['features']:
            X_upscale_reg.append(subset_cube[var_name])
        
        X_upscale_reg_flattened = []

        for arr in X_upscale_reg:
            data = arr.data.flatten()
            X_upscale_reg_flattened.append(data)
            
        X_upscale_reg_flattened = da.array(X_upscale_reg_flattened).transpose().compute()
        
        EI_pred= np.zeros(X_upscale_reg_flattened.shape[0]) * np.nan
        
        mask = (np.all(np.isfinite(X_upscale_reg_flattened), axis=1))
        
        if (X_upscale_reg_flattened[mask].shape[0]>0):
            
            average_distance = self.calculate_distance(best_KNN, scaler, X_upscale_reg_flattened[mask])
            #epsilon = calculate_epsilon(self.x_train, self.y_pred, self.y_train)
            #average_distance_weighted = average_distance * epsilon
            
            EI_pred[mask] = average_distance
            EI_pred = EI_pred.reshape(len(subset_cube.latitude), len(subset_cube.longitude), len(subset_cube.time))
            
            output_EI_xr = xr.DataArray(EI_pred, 
                                        coords={"latitude": subset_cube.latitude, 
                                                "longitude": subset_cube.longitude,
                                                "time": subset_cube.time}, 
                                        dims=["latitude", "longitude", "time"]).to_dataset(name="Extrapolation_Index")
                  
            self.interpolation_cube.update_cube(output_EI_xr, initialize=False)
    
    def calculate_global_index(self) -> None:
        
        self.interpolation_cube = DataCube(cube_config = self.cube_config)
        self.interpolation_cube.init_variable(self.cube_config['cube_variables'], 
                                              njobs= len(self.cube_config['cube_variables'].keys()))            
        
        LatChunks = np.array_split(self.interpolation_cube.cube.latitude.values, self.cube_config["num_chunks"])
        LonChunks = np.array_split(self.interpolation_cube.cube.longitude.values, self.cube_config["num_chunks"])
        
        AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                       "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
       
        self.x_train = xr.open_dataset(self.DataConfig['training_dataset'])[self.DataConfig['features']]
        self.y_train = xr.open_dataset(self.DataConfig['training_dataset'])[self.DataConfig['target']]
        
        #weights = self.calculate_weights(self.x_train, self.y_train)
        best_KNN, scaler = self.trainKNN(self.x_train, self.y_train)
        
        for tree_cover in self.cube_config["tree_cover_tresholds"]:
            self.tree_cover = tree_cover
            if (self.n_jobs > 1):
                with dask.config.set({'distributed.worker.memory.target': 50*1024*1024*1024, 
                                      'distributed.worker.threads': 2}):

                    futures = [self.calculate_index(i, best_KNN, scaler) for i in AllExtents]
                    dask.compute(*futures, num_workers=self.n_jobs)    
            else:
                for extent in AllExtents:
                    self.calculate_index(extent).compute()
            
            
