#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File    :   ml_dataloader.py

This module provides functionalities for generating dataloaders for machine learning models.

Example usage:
--------------
from ml_dataloader import MLDataModule

# Create an MLDataModule instance
data_config = {...}
target = {...}
features = {...}
train_subset = {...}
valid_subset = {...}
test_subset = {...}
norm_stats = {...}

ml_data_module = MLDataModule(
    method='MLPRegressor',
    DataConfig=data_config,
    target=target,
    features=features,
    train_subset=train_subset,
    valid_subset=valid_subset,
    test_subset=test_subset,
    normalize=True,
    norm_stats=norm_stats
)

train_loader = ml_data_module.train_dataloader()
valid_loader = ml_data_module.val_dataloader()
test_loader = ml_data_module.test_dataloader()
"""
from typing import Any

import xarray as xr

from ageUpscaling.dataloaders.base import MLData

class MLDataModule(MLData):
    
    """
    Define dataloaders.

    Parameters
    ----------
    method : str
        The method used for machine learning.
    DataConfig : dict[str, Any]
        The data configuration.
    target : dict[str, Any]
        The target variable for the machine learning model.
    features : dict[str, Any]
        The features used for the machine learning model.
    train_subset : dict[str, Any]
        Training set selection.
    valid_subset : dict[str, Any]
        Same as `train_subset`, for validation set.
    test_subset : dict[str, Any]
        Same as `train_subset`, for test set.
    normalize : bool, optional
        Whether to normalize the data. Default is False.
    norm_stats : dict[str, dict[str, float]], optional
        The normalization statistics for the data. Default is an empty dictionary.
    **kwargs : keyword arguments
        Additional keyword arguments passed to `DataLoader`.
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

        for var in  self.target + self.features:
            data = xr.open_dataset(self.DataConfig['training_dataset']).sel(cluster = train_subset)[var]
            data_mean = data.mean().compute().item()
            data_std = data.std().compute().item()
            self.norm_stats[var] = {'mean': data_mean, 'std': data_std}

    def train_dataloader(self) -> MLData:
        """
        Returns a dataloader for the training set.

        Returns
        -------
        MLData
            Dataloader for the training set, containing the features and target values for the training set.
        """
        train_data = MLData(self.method,
                            self.DataConfig, 
                            self.target, 
                            self.features, 
                            self.train_subset,
                            self.normalize,
                            self.norm_stats,
                            training=False)        
            
        return train_data

    def val_dataloader(self) -> MLData:
        """
        Returns a dataloader for the validation set.

        Returns
        -------
        MLData
            Dataloader for the validation set, containing the features and target values for the validation set.
        """
        valid_data = MLData(self.method, 
                            self.DataConfig, 
                            self.target, 
                            self.features, 
                            self.valid_subset, 
                            self.normalize,
                            self.norm_stats,
                            training=False)
            
        return valid_data  

    def test_dataloader(self) -> MLData:
        """
        Returns a dataloader for the test set.
    
        Returns
        -------
        MLData
            Dataloader for the test set, containing the features and target values for the test set.
        """
        test_data = MLData(self.method,
                           self.DataConfig,
                           self.target, 
                           self.features, 
                           self.test_subset, 
                           self.normalize,
                           self.norm_stats,
                           training=False)
            
        return test_data

    
