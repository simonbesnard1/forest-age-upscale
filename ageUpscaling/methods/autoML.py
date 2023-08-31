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
import numpy as np
from typing import Any
import os

import xarray as xr
import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor

from ageUpscaling.dataloaders.ml_dataloader import MLDataModule

class AutoML:
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
                 study_dir: str=None,
                 DataConfig:dict=None,
                 method:str = 'AutoMLRegressor') -> None:
        
        self.study_dir = study_dir
        
        self.tune_dir = os.path.join(study_dir, "tune/{method}".format(method= method))
        if not os.path.exists(self.tune_dir):
            os.makedirs(self.tune_dir)
            
        self.DataConfig = DataConfig
        self.method = method
            
        self.DataConfig = DataConfig
        self.method = method
        
    def get_datamodule(self, 
                       method:str = 'AutoMLRegressor',
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
              oversampling:bool= True,
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

        self.mldata = self.get_datamodule(method= self.method,
                                          DataConfig=self.DataConfig, 
                                          target=self.DataConfig['target'],
                                          features = self.DataConfig['features_selected'],
                                          train_subset=train_subset,
                                          valid_subset=valid_subset,
                                          test_subset=test_subset)
          
        train_data = self.mldata.train_dataloader().get_xy()
        
        if oversampling and self.method == "AutoMLRegressor":
            
            age_classes, current_points = np.unique(np.round(train_data['target'], -1), return_counts=True)
            desired_points_per_class = np.nanmax(current_points)
            
            Y_sample = []
            X_sample = []
            for a, b in zip(age_classes, current_points):
                
                required_samples = desired_points_per_class - b
                
                if required_samples > 0:
                    idx_ =  np.where(np.round(train_data['target'], -1) == a)[0]   
                    idx_sample = np.random.choice(idx_, required_samples)
                    Y_sample.append(train_data['target'][idx_sample]), 
                    X_sample.append(train_data['features'][idx_sample])
                    
            Y_sample = np.concatenate(Y_sample)
            X_sample = np.concatenate(X_sample)
            
            train_data=  pd.DataFrame(np.concatenate([np.concatenate([train_data['target'], Y_sample]).reshape(-1, 1),                                         
                                         np.concatenate([train_data['features'], X_sample])], axis=1), columns =self.DataConfig['target']+ self.DataConfig['features_selected'])
        
        else:        
            train_data = pd.DataFrame(np.concatenate([train_data['target'].reshape(-1, 1), train_data['features']], axis=1), columns =self.DataConfig['target']+ self.DataConfig['features_selected'])
        
        val_data = self.mldata.val_dataloader().get_xy()
        val_data = pd.DataFrame(np.concatenate([val_data['target'].reshape(-1, 1), val_data['features']], axis=1), columns =self.DataConfig['target'] + self.DataConfig['features_selected']) 
        
        if self.method == "AutoMLRegressor":
            self.best_model = TabularPredictor(self.DataConfig['target'][0], eval_metric='mean_squared_error', problem_type ='regression', 
                                               path = self.tune_dir).fit(train_data = TabularDataset(train_data), tuning_data = TabularDataset(val_data),
                                                                                  presets='best_quality', use_bag_holdout=True)
        elif self.method == "AutoMLClassifier":
            self.best_model = TabularPredictor(self.DataConfig['target'][0], eval_metric='log_loss', problem_type ='binary', 
                                               path = self.tune_dir).fit(TabularDataset(train_data), tuning_data = TabularDataset(val_data),
                                                                         presets='best_quality', use_bag_holdout=True)                                                            
                                                                         
                                                                         
    def predict_clusters(self, 
                        save_cube:str) -> None:
        """Predict the target variables for the given data and save the results in a prediction zarr cube.
    
        Parameters:
            save_cube: str
                Path to the output netCDF file where the predictions will be saved.
        """
        
        X = self.mldata.test_dataloader().get_x(method= self.method, features= self.DataConfig['features_selected'])
        Y = self.mldata.test_dataloader().get_y(target= self.DataConfig['target'], 
                                               method= self.method, 
                                               max_forest_age= self.DataConfig['max_forest_age'])

        for cluster_ in np.arange(len(self.mldata.test_subset)):
            X_cluster = X[cluster_, : , :].reshape(-1, len(self.DataConfig['features_selected']))
            Y_cluster = Y[:, cluster_, :].reshape(-1)
            mask_nan = (np.all(np.isfinite(X_cluster), axis=1)) & (np.isfinite(Y_cluster))
            if X_cluster[mask_nan, :].shape[0]>0:
                dpred = pd.DataFrame(np.concatenate([Y_cluster[mask_nan,].reshape(-1, 1), X_cluster[mask_nan, :]], axis=1), columns =self.DataConfig['target'] + self.DataConfig['features_selected']) 
                y_hat = self.best_model.predict(dpred).values
                preds = xr.Dataset()
                
                if self.method == "AutoMLClassifier": 
                    out_var = 'oldGrowth'
                elif self.method == "AutoMLRegressor": 
                    out_var = 'forestAge'
                
                preds["{out_var}_pred".format(out_var = out_var)] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["{out_var}_obs".format(out_var = out_var)] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                
                save_cube.CubeWriter(preds.transpose('sample', 'cluster'))
   
    
    