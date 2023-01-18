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
from abc import ABC
from itertools import product

import numpy as np
import yaml as yml

import multiprocessing as mp

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV,  KFold

from ageUpscaling.core.cube import DataCube

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
                 DataConfig_path: str,
                 cube_config_path: str,            
                 base_dir: str,
                 algorithm: str = 'MLP',
                 n_jobs: int = 1,
                 **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
        
        self.algorithm = algorithm
        self.base_dir = base_dir
        self.n_jobs = n_jobs
        self.cube_config['cube_location'] = os.path.join(self.study_dir, self.cube_config['cube_name'])
        
        @staticmethod
        def calculate_distance(x_train, 
                              y_train, 
                              x_test, 
                              weights, 
                              metric='euclidean', 
                              k_range = range(1, 31)):
            
           param_grid = dict(n_neighbors=k_range)
           knn = KNeighborsRegressor(metric=metric)
           kf = KFold(n_splits=10)
           grid = GridSearchCV(knn, param_grid, cv=kf, scoring='neg_mean_squared_error')
           grid.fit(x_train, y_train)
           distances, indices = grid.best_estimator_.kneighbors(x_test)
           distance_sum = 0
           for i in range(grid.best_params_["n_neighbors"]):
               distance_sum += np.sum(weights * np.abs(x_test - x_train[indices[0][i]]))
           average_distance = distance_sum / grid.best_params_["n_neighbors"]
          
           return average_distance

        @staticmethod
        def calculate_epsilon(x_train, 
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
        
        @staticmethod
        def calculate_weights(X, 
                              Y):

            weights = X + Y
            
            return weights   
        
        def calculate_index(self):
             
            weights = calculate_weights(self.x_train, self.y_train)
            average_distance = calculate_distance(self.x_train, self.y_train, self.x_test, weights)
            #epsilon = calculate_epsilon(self.x_train, self.y_pred, self.y_train)
            #average_distance_weighted = average_distance * epsilon
            average_distance_weighted = average_distance
            
            return average_distance_weighted
        
        def calculate_global_index(self):
            
            self.index_cube = DataCube(cube_config = self.cube_config)

            LatChunks = np.array_split(self.index_cube.cube.latitude.values, self.cube_config["num_chunks"])
            LonChunks = np.array_split(self.index_cube.cube.longitude.values, self.cube_config["num_chunks"])
            
            AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                           "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                        for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]

            if(self.n_jobs > 1):
                
                p=mp.Pool(self.n_jobs, maxtasksperchild=1)
                p.map(self.calculate_index, 
                      AllExtents)
                p.close()
                p.join()
            else:
                _ = map(self.calculate_index, AllExtents)
            
