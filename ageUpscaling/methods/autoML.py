#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   RF.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for training RF model
"""
import os
import numpy as np
from typing import Any

import xarray as xr

from tpot import TPOTRegressor
from tpot import TPOTClassifier

from ageUpscaling.dataloaders.ml_dataloader import MLDataModule
from ageUpscaling.methods.feature_selection import FeatureSelection

class TPOT:
    """A method class for training and evaluating an autoML models.
    
    Parameters
    ----------
    tune_dir : str, default is None
        Directory to save the model experiment. If not provided, the model experiment will not be saved.
    DataConfig : dict, default is None
        Dictionary containing the data configuration.
    method : str, default is 'TPOTRegressor'
        String defining the type of RF model to use. Can be 'TPOTRegressor' for a regression model or 'TPOTClassifier' for a classification model.
    """
    
    def __init__(self,
                 tune_dir: str=None,
                 DataConfig:dict=None,
                 method:str = 'TPOTRegressor') -> None:

        self.tune_dir = tune_dir
        
        if not os.path.exists(tune_dir):
            os.makedirs(tune_dir)
            
        self.DataConfig = DataConfig
        self.method = method
        
    def get_datamodule(self, 
                       method:str = 'TPOTRegressor',
                       DataConfig: dict[str, Any] = {},
                       target: dict[str, Any] = {},
                       features: dict[str, Any] = {},
                       train_subset: dict[str, Any] = {},
                       valid_subset: dict[str, Any] = {},
                       test_subset: dict[str, Any] = {},
                       **kwargs) -> MLDataModule:
        """Returns the data module for training the model.

        Parameters:
            method: str, default is 'TPOTRegressor'
                The type of model to use for training ('TPOTRegressor' or 'TPOTClassifier').
            DataConfig: dict[str, Any]
                The data configuration.
            target: dict[str, Any]
                The target variables and their configuration.
            features: dict[str, Any]
                The features and their configuration.
            train_subset: dict[str, Any]
                The subset of the data to use for training.
            valid_subset: dict[str, Any]
                The subset of the data to use for validation.
            test_subset: dict[str, Any]
                The subset of the data to use for testing.
            **kwargs:
                Additional keyword arguments to pass to the `MLDataModule` constructor.
    
        Returns:
            mlData: MLDataModule
                The data module for training the model.
        """
        
        mlData = MLDataModule(method,
                              DataConfig,
                              target,
                              features,
                              train_subset, 
                              valid_subset, 
                              test_subset)

        return mlData
        
    def train(self,  
              train_subset:dict={},
              valid_subset:dict={}, 
              test_subset:dict={},
              feature_selection:bool= False,
              feature_selection_method:str="recursive", 
              n_jobs:int=10) -> None:
        
        """Trains an RF model using the specified training and validation datasets.

        Parameters:
            train_subset: dict
                Dictionary containing the training dataset.
            valid_subset: dict
                Dictionary containing the validation dataset.
            test_subset: dict
                Dictionary containing the test dataset.
            feature_selection: bool, optional
                If True, performs feature selection on the training data before training the model.
            feature_selection_method: str, optional
                Method to use for feature selection. Must be one of "boruta" or "recursive".
            n_jobs: int, optional
                Number of jobs to use when fitting the model.
        """

        if feature_selection:
            mldata_feature_sel = self.get_datamodule(method= self.method,
                                              DataConfig=self.DataConfig, 
                                              target=self.DataConfig['target'],
                                              features =  self.DataConfig['features'],
                                              train_subset=train_subset,
                                              valid_subset=valid_subset,
                                              test_subset=test_subset)            
            features_selected = FeatureSelection(method=self.method, 
                                                 feature_selection_method = feature_selection_method, 
                                                 features = self.DataConfig['features']).get_features(data = mldata_feature_sel.train_dataloader().get_xy())
        
        self.final_features = [features_selected if feature_selection else self.DataConfig['features']][0]
        
        self.mldata = self.get_datamodule(method= self.method,
                                          DataConfig=self.DataConfig, 
                                          target=self.DataConfig['target'],
                                          features = self.DataConfig['features'],
                                          train_subset=train_subset,
                                          valid_subset=valid_subset,
                                          test_subset=test_subset)
          
        train_data = self.mldata.train_dataloader().get_xy()
        
        if self.method == "TPOTRegressor":
            model = TPOTRegressor(n_jobs=n_jobs)
        elif self.method == "TPOTClassifier":
            model = TPOTClassifier(n_jobs=n_jobs)
        
        model.fit(train_data['features'], train_data['target'])

        self.best_model = model
    
    def predict_clusters(self, 
                        save_cube:str) -> None:
        """Predict the target variables for the given data and save the results in a prediction zarr cube.
    
        Parameters:
            save_cube: str
                Path to the output netCDF file where the predictions will be saved.
        """
        
        X = self.mldata.test_dataloader().get_x(method= self.method, features= self.DataConfig['features'])
        Y = self.mldata.test_dataloader().get_y(target= self.DataConfig['target'], 
                                               method= self.method, 
                                               max_forest_age= self.DataConfig['max_forest_age'])

        for cluster_ in np.arange(len(self.mldata.test_subset)):
            X_cluster = X[cluster_, : , :].reshape(-1, len(self.DataConfig['features']))
            Y_cluster = Y[:, cluster_, :].reshape(-1)
            mask_nan = (np.all(np.isfinite(X_cluster), axis=1)) & (np.isfinite(Y_cluster))
            if X_cluster[mask_nan, :].shape[0]>0:
                y_hat = self.best_model.predict(X_cluster[mask_nan, :])
                preds = xr.Dataset()
                
                if self.method == "TPOTClassifier": 
                    out_var = 'oldGrowth'
                elif self.method == "TPOTRegressor": 
                    out_var = 'forestAge'
                
                preds["{out_var}_pred".format(out_var = out_var)] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["{out_var}_obs".format(out_var = out_var)] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                
                save_cube.update_cube(preds.transpose('sample', 'cluster'), initialize=True)
   
    
    