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
import pickle
from typing import Any

import xarray as xr

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, roc_auc_score


import optuna

from ageUpscaling.dataloaders.ml_dataloader import MLDataModule
#from ageUpscaling.utils.metrics import mef_gufunc

class RandomForest:
    """A method class for training and evaluating an RandomForest model.
    
    Parameters
    ----------
    tune_dir : str, default is None
        Directory to save the model experiment. If not provided, the model experiment will not be saved.
    DataConfig : dict, default is None
        Dictionary containing the data configuration.
    method : str, default is 'RandomForestRegressor'
        String defining the type of RF model to use. Can be 'RandomForestRegressor' for a regression model or 'RandomForestClassifier' for a classification model.
    """
    
    def __init__(self,
                 study_dir: str=None,
                 DataConfig:dict=None,
                 method:str = 'RandomForestRegressor') -> None:

        self.study_dir = study_dir
        
        self.tune_dir = os.path.join(study_dir, "tune")
        if not os.path.exists(self.tune_dir):
            os.makedirs(self.tune_dir)
            
        self.DataConfig = DataConfig
        self.method = method
        
    def get_datamodule(self, 
                       method:str = 'RandomForestRegressor',
                       DataConfig: dict[str, Any] = {},
                       target: dict[str, Any] = {},
                       features: dict[str, Any] = {},
                       train_subset: dict[str, Any] = {},
                       valid_subset: dict[str, Any] = {},
                       test_subset: dict[str, Any] = {},
                       **kwargs) -> MLDataModule:
        """Returns the data module for training the model.

        Parameters:
            method: str, default is 'RandomForestRegressor'
                The type of model to use for training ('RandomForestRegressor' or 'RandomForestClassifier').
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
              n_jobs:int=10) -> None:
        
        """Trains an RF model using the specified training and validation datasets.

        Parameters:
            train_subset: dict
                Dictionary containing the training dataset.
            valid_subset: dict
                Dictionary containing the validation dataset.
            test_subset: dict
                Dictionary containing the test dataset.
            n_jobs: int, optional
                Number of jobs to use when fitting the model.
        """

        self.mldata = self.get_datamodule(method= self.method,
                                          DataConfig=self.DataConfig, 
                                          target=self.DataConfig['target'],
                                          features = self.DataConfig['features_selected'],
                                          train_subset= np.concatenate([train_subset, valid_subset]),
                                          valid_subset=valid_subset,
                                          test_subset=test_subset)
          
        train_data = self.mldata.train_dataloader().get_xy()
        val_data = self.mldata.val_dataloader().get_xy()
                    
        if not os.path.exists(self.tune_dir + '/trial_model/'):
            os.makedirs(self.tune_dir + '/trial_model/')
        
        study = optuna.create_study(study_name = 'hpo_ForestAge', 
                                    #storage='sqlite:///' + self.tune_dir + '/trial_model/hp_trial.db',
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                    reduction_factor=2, 
                                                                                    min_early_stopping_rate=10),
                                    direction=['minimize' if self.method == 'RandomForestRegressor' else 'maximize'][0])
        study.optimize(lambda trial: self.hp_search(trial, train_data, val_data, self.DataConfig, self.tune_dir), 
                        n_trials=self.DataConfig['hyper_params']['number_trials'], n_jobs=n_jobs)
        
        with open(self.tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = study.best_trial.number), "rb") as fin:
            self.best_model = pickle.load(fin)            
            
    def hp_search(self, 
                   trial: optuna.Trial,
                   train_data:dict,
                   val_data:dict,
                   DataConfig:dict,
                   tune_dir:str,
                   oversampling:bool= True) -> float:
        """Searches for the optimal hyperparameters for the machine learning model.
        
        Parameters
        ----------
        trial: optuna.Trial
            The trial object for the hyperparameter optimization.
        train_data: dict
            A dictionary containing the training data.
        val_data: dict
            A dictionary containing the validation data.
        DataConfig: dict
            A dictionary containing the data configuration.
        tune_dir: str
            The directory to save the model experiment.
            
        Returns
        -------
        float
            The loss of the model.
        """
        
        hyper_params = {"n_estimators": trial.suggest_int('n_estimators', DataConfig['hyper_params']['n_estimators']['min'], DataConfig['hyper_params']['n_estimators']['max'], step=DataConfig['hyper_params']['n_estimators']['step']),
                        "max_depth": trial.suggest_int('max_depth', DataConfig['hyper_params']['max_depth']['min'], DataConfig['hyper_params']['max_depth']['max'], step=DataConfig['hyper_params']['max_depth']['step']),
                        "min_samples_split": trial.suggest_int('min_samples_split', DataConfig['hyper_params']['min_samples_split']['min'], DataConfig['hyper_params']['min_samples_split']['max'], step=DataConfig['hyper_params']['min_samples_split']['step']),
                        "min_samples_leaf": trial.suggest_int('min_samples_leaf', DataConfig['hyper_params']['min_samples_leaf']['min'], DataConfig['hyper_params']['min_samples_leaf']['max'], step=DataConfig['hyper_params']['min_samples_leaf']['step'])}
                                        
        if self.method == "RandomForestRegressor":
            model_ = RandomForestRegressor(**hyper_params, n_jobs=10, oob_score=True, bootstrap=True)
            
        elif self.method == "RandomForestClassifier":
            model_ = RandomForestClassifier(**hyper_params, class_weight = "balanced", n_jobs=10, oob_score=True, bootstrap=True)
          
        if oversampling and self.method == "RandomForestRegressor":
            
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
           model_.fit(np.concatenate([train_data['features'], X_sample]) , np.concatenate([train_data['target'], Y_sample]))
        
        else: 
           model_.fit(train_data['features'], train_data['target'])
        
        with open(tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return model_.oob_score_
    
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
                y_hat = self.best_model.predict(X_cluster[mask_nan, :])
                preds = xr.Dataset()
                
                if self.method == "RandomForestClassifier": 
                    out_var = 'oldGrowth'
                elif self.method == "RandomForestRegressor": 
                    out_var = 'forestAge'
                
                preds["{out_var}_pred".format(out_var = out_var)] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["{out_var}_obs".format(out_var = out_var)] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                
                save_cube.CubeWriter(preds.transpose('sample', 'cluster'))
   
    
    