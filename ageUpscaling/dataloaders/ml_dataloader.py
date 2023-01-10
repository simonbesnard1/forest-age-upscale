#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   ml_dataloader.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for generating dataloaders
"""
from typing import Any

import xarray as xr

from ageUpscaling.dataloaders.base import MLData

class MLDataModule(MLData):
    
    """Define dataloaders.

    Parameters:
        cube_path: str
            Path to the datacube.
        DataConfig: DataConfig
            The data configuration.
        train_subset: Dict[str, Any]:
            Training set selection.
        valid_subset: Dict[str, Any]:
            Same as `train_subset`, for validation set.
        test_subset: Dict[str, Any]:
            Same as `train_subset`, for test set.
        **kwargs:
            Keyword arguments passed to `DataLoader`, same for training, validation and test set loader.
    """
    
    def __init__(self,
                 method:str='MLPRegressor',
                 DataConfig: dict[str, Any] = {},
                 target: dict[str, Any] = {},
                 features: dict[str, Any] = {},            
                 train_subset: dict[str, Any] = {},
                 valid_subset: dict[str, Any] = {},
                 test_subset: dict[str, Any] = {},
                 normalize:bool= False,
                 norm_stats: dict[str, dict[str, float]] = {},
                 **kwargs) -> None:
        
        super().__init__()

        self.method = method
        self.DataConfig = DataConfig
        self.target = target
        self.features = features        
        self.train_subset = train_subset
        self.valid_subset = valid_subset
        self.test_subset = test_subset
        self.normalize = normalize
        self.norm_stats = norm_stats
        self._kwargs = kwargs
        
        if len(self.norm_stats) == 0:

            for var in  self.target + self.features:
                data = xr.open_dataset(self.DataConfig['training_dataset']).sel(cluster = train_subset)[var]
                data_mean = data.mean().compute().item()
                data_std = data.std().compute().item()
                self.norm_stats[var] = {'mean': data_mean, 'std': data_std}

    def train_dataloader(self) -> MLData:
        """Returns a dataloader for the training set.

        Returns:
            MLData: Dataloader for the training set, containing the features and target values for the training set.
        """
        
        train_data = MLData(self.method, 
                            self.DataConfig, 
                            self.target, 
                            self.features, 
                            self.train_subset,
                            self.normalize,
                            self.norm_stats)        
            
        return train_data

    def val_dataloader(self) -> MLData:
        """Returns a dataloader for the validation set.

        Returns:
            MLData: Dataloader for the validation set, containing the features and target values for the validation set.
        """
        
        valid_data = MLData(self.method, 
                            self.DataConfig, 
                            self.target, 
                            self.features, 
                            self.valid_subset, 
                            self.normalize,
                            self.norm_stats)
            
        return valid_data  

    def test_dataloader(self) -> MLData:
        """Returns a dataloader for the test set.

        Returns:
            MLData: Dataloader for the test set, containing the features and target values for the test set.
        """

        test_data = MLData(self.method, 
                           self.DataConfig,
                           self.target, 
                           self.features, 
                           self.test_subset, 
                           self.normalize,
                           self.norm_stats)
            
        return test_data

    
