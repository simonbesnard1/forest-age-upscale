#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:30:04 2022

@author: sbesnard
"""
from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np
from typing import Any
import xarray as xr

class MLData:
    """A dataset defines how samples are generated.
    
    Parameters:
        cube_path: str
            Path to the datacube.
        data_config: dict
            The data configuration.
        subset: Dict[str, Any]:
            Subset selection.
    """
    def __init__(
        self,
        cube_path:str, 
        subset: dict[str, Any] = {}, 
        data_config: dict[str, Any] = {},
        norm_stats: dict[str, dict[str, float]] = {}):
    
        super().__init__()
        
        self.cube_path = cube_path
        self.subset = subset
        self.data_config = data_config
        self.norm_stats = norm_stats
                
    def get_x(self,
              features:dict):

        """
        Method to concatenate the features 

        Return
        ------
        x (np.array): the concatenated features 
        """
        
        data = xr.open_dataset(self.cube_path).sel(cluster = self.subset)

        X = data[features]
        
        X = self.norm(X, self.norm_stats)
        
        return X.to_array().transpose('cluster','sample', 'variable').values

    def get_y(self,
              target,
              method,
              max_forest_age)-> Tuple[ArrayLike, ArrayLike, int]:
        
        """
        Method to get the target.

        Return
        ------
        y (np.array): the target 
        """
        Y = xr.open_dataset(self.cube_path).sel(cluster = self.subset)[target]
        
        if method == 'MLPClassifier':
            Y = Y.to_array().values
            mask_old = Y== max_forest_age
            mask_young = Y<max_forest_age
            Y[mask_old] = 1
            Y[mask_young] = 0    
        
        elif method == 'MLPRegressor':
            Y = Y.where(Y<max_forest_age).to_array().values
        
        return Y#.reshape(-1)
            
    def get_xy(self,  
               standardize:bool=True):
        
        self.y = self.get_y(target=self.data_config['target'], 
                            method = self.data_config['method'][0], 
                            max_forest_age =self.data_config['max_forest_age'][0]).reshape(-1)
        
        self.x = self.get_x(features= self.data_config['features']).reshape(-1, len(self.data_config['features']))
        
        mask_nan = np.isfinite(self.y)
        self.x, self.y = self.x[mask_nan, :], self.y[mask_nan]    
        
        return {'features' : self.x.astype('float32'), "target": self.y.astype('float32'), 'norm_stats': self.norm_stats}
    
    def norm(self, 
             x: xr.Dataset, 
             norm_stats: dict) -> xr.Dataset:
        
        for var in x.data_vars:
            x[var] = (x[var] - norm_stats[var]['mean']) / norm_stats[var]['std']

        return x
    