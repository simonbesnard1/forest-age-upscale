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
        data_config: dict[str, Any] = {}):
    
        super().__init__()
        
        self.cube_path = cube_path
        self.subset = subset
        self.data_config = data_config
                
    def get_x(self,
              features:dict,
              standardize:bool):

        """
        Method to concatenate the features 

        Return
        ------
        x (np.array): the concatenated features 
        """
        
        data = xr.open_dataset(self.cube_path).sel(spatial_cluster = self.subset)

        X = data[features]
        
        if standardize:
            for var_ in list(X.keys()):
                X[var_] = self.standardize(X[var_])
        
        return X.to_array().transpose('plot', 'sample', 'variable').values

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
        Y = np.round(xr.open_dataset(self.cube_path).sel(spatial_cluster = self.subset)[target])
        
        if method == 'MLPClassifier':
            Y = Y.to_array().values
            mask_old = Y== max_forest_age
            mask_young = Y<max_forest_age
            Y[mask_old] = 1
            Y[mask_young] = 0    
        
        elif method == 'MLPRegressor':
            Y = Y.where(Y<max_forest_age).to_array().values
        
        return Y
    
    def standardize(self, x):
        return (x - np.nanmean(x)) / np.nanstd(x)
        
    def get_xy(self, 
               standardize:bool=True):
        
        self.y = self.get_y(target=self.data_config['target'], 
                            method = self.data_config['method'][0], 
                            max_forest_age =self.data_config['max_forest_age'][0]).reshape(-1)
        self.x = self.get_x(features= self.data_config['features'],
                            standardize= standardize).reshape(-1, len(self.data_config['features']))
        mask_nan = (np.all(np.isfinite(self.x), axis=1)) & (np.isfinite(self.y))
        self.x, self.y = self.x[mask_nan, :], self.y[mask_nan]
         
        return {'features' : self.x.astype('float32'), "target": self.y.astype('int')}
    