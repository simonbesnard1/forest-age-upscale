#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   base.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for defining how samples are generated
"""
from typing import Tuple, Any
from numpy.typing import ArrayLike
import numpy as np

import xarray as xr

from abc import ABC

class MLData(ABC):
    """An abstract class defining a dataset used for machine learning.

    A dataset defines how samples are generated.
    
    Parameters:
        method: str
            The method used for machine learning.
        DataConfig: dict
            The data configuration containing information about the data.
        target: dict[str, Any]
            The target variable for the machine learning model.
        features: dict[str, Any]
            The features used for the machine learning model.
        subset: dict[str, Any]
            A subset selection of the data to use for training and validation.
        norm_stats: dict[str, dict[str, float]]
            The normalization statistics for the data.
    """
    def __init__(self,
                 method:str='MLPRegressor',
                 DataConfig: dict[str, Any] = {},
                 target: dict[str, Any] = {},
                 features: dict[str, Any] = {},     
                 subset: dict[str, Any] = {}, 
                 norm_stats: dict[str, dict[str, float]] = {}):
    
        super().__init__()
        
        self.DataConfig = DataConfig
        self.subset = subset
        self.target = target
        self.features = features 
        self.norm_stats = norm_stats
        self.method = method
                
    def get_x(self,
              features:dict) -> Tuple[ArrayLike, ArrayLike, float]:

        """Concatenate the features and normalize them.

        Parameters
        ----------
        features : dict
            Dictionary containing the features to be concatenated.
    
        Returns
        -------
        x : ArrayLike
            The concatenated and normalized features.    
        """
        
        data = xr.open_dataset(self.DataConfig['training_dataset']).sel(cluster = self.subset)

        X = data[features]
        
        X = self.norm(X, self.norm_stats)
        
        return X.to_array().transpose('cluster','sample', 'variable').values

    def get_y(self,
              target: str,
              method: str,
              max_forest_age: int) -> Tuple[ArrayLike, ArrayLike, int]:
        """Get the target data for the given method and maximum forest age.
    
        Parameters
        ----------
        target: str
            The name of the target variable to retrieve from the training dataset.
        method: str
            The method to be used for training (either 'MLPClassifier' or 'MLPRegressor').
        max_forest_age: int
            The maximum age of forests to consider.
    
        Returns
        -------
        y: np.array
            The target data, transformed as necessary for the given method and maximum forest age.
        """

        Y = xr.open_dataset(self.DataConfig['training_dataset']).sel(cluster = self.subset)[target]
        
        if method == 'MLPClassifier':
            Y = Y.to_array().values
            mask_old = Y==max_forest_age
            mask_young = Y<max_forest_age
            Y[mask_old] = 1
            Y[mask_young] = 0    
        
        elif method == 'MLPRegressor':
            Y = Y.where(Y<max_forest_age)
            Y = self.norm(Y, self.norm_stats).to_array().values
        return Y
            
    def get_xy(self,  
               standardize:bool=True) -> dict:
        """Get features and target arrays from the dataset.
    
        Parameters
        ----------
        standardize : bool
            Whether to standardize the features.
    
        Returns
        -------
        dict
            A dictionary containing the features and target arrays as well as the normalization statistics.
        """
        
        self.y = self.get_y(target=self.target, 
                            method = self.method, 
                            max_forest_age =self.DataConfig['max_forest_age'][0]).reshape(-1)
        
        self.x = self.get_x(features= self.features).reshape(-1, len(self.features))        
        mask_nan = (np.all(np.isfinite(self.x), axis=1)) & (np.isfinite(self.y))
        
        self.x, self.y = self.x[mask_nan, :], self.y[mask_nan]    
        if self.method == 'MLPRegressor': 
            self.y=self.y.astype('float32')
        elif self.method == 'MLPClassifier': 
            self.y=self.y.astype('int8')
        
        return {'features' : self.x.astype('float32'), "target": self.y, 'norm_stats': self.norm_stats}
    
    def norm(self, 
             x: xr.Dataset, 
             norm_stats: dict) -> xr.Dataset:
        """Normalize the data in the given xarray dataset using the provided normalization statistics.
        
        Parameters
        ----------
        x : xr.Dataset
            The xarray dataset to normalize.
        norm_stats : dict
            A dictionary containing the normalization statistics for each data variable in the dataset.
            The dictionary should have the format {variable_name: {'mean': mean_value, 'std': std_value}}.
        
        Returns
        -------
        xr.Dataset
            The normalized xarray dataset.
        """
        
        for var in x.data_vars:
            x[var] = (x[var] - norm_stats[var]['mean']) / norm_stats[var]['std']

        return x
    
    