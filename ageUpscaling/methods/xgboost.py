#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   xgboost.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for training Xgboost model
"""
import os
import numpy as np
import pickle
from typing import Any

import xarray as xr

import xgboost as xgb

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.INFO)

from ageUpscaling.dataloaders.ml_dataloader import MLDataModule


class XGBoost:
    """A method class for training and evaluating an XGBoost model.
    
    Parameters
    ----------
    tune_dir : str, default is None
        Directory to save the model experiment. If not provided, the model experiment will not be saved.
    DataConfig : dict, default is None
        Dictionary containing the data configuration.
    method : str, default is 'XGBoostRegressor'
        String defining the type of XGBoost model to use. Can be 'XGBoostRegressor' for a regression model or 'XGBoostClassifier' for a classification model.
    """
    
    def __init__(self,
                 study_dir: str=None,
                 DataConfig:dict=None,
                 method:str = 'XGBoostRegressor') -> None:

        self.study_dir = study_dir
        
        self.tune_dir = os.path.join(study_dir, "tune/{method}".format(method= method))
        if not os.path.exists(self.tune_dir):
            os.makedirs(self.tune_dir)
        
        self.DataConfig = DataConfig
        self.method = method
        
    def get_datamodule(self, 
                       method:str = 'XGBoostRegressor',
                       DataConfig: dict[str, Any] = {},
                       target: dict[str, Any] = {},
                       features: dict[str, Any] = {},
                       train_subset: dict[str, Any] = {},
                       valid_subset: dict[str, Any] = {},
                       test_subset: dict[str, Any] = {},
                       **kwargs) -> MLDataModule:
        """Returns the data module for training the model.

        Parameters:
            method: str, default is 'XGBoostRegressor'
                The type of model to use for training ('XGBoostRegressor' or 'XGBoostClassifier').
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
        
        """Trains an XGBoost model using the specified training and validation datasets.

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
                                          train_subset=train_subset,
                                          valid_subset=valid_subset,
                                          test_subset=test_subset)
          
        train_data = self.mldata.train_dataloader().get_xy()
        val_data = self.mldata.val_dataloader().get_xy()
                    
        if not os.path.exists(self.tune_dir + '/trial_model/'):
            os.makedirs(self.tune_dir + '/trial_model/')
        
        
        study = optuna.create_study(study_name = 'hpo_ForestAge', 
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                    reduction_factor=2, 
                                                                                    min_early_stopping_rate=10),
                                    direction=["minimize" if self.method == 'XGBoostRegressor' else 'minimize'][0])
                
        num_trials = self.DataConfig['hyper_params']['number_trials']

        max_workers = min(num_trials, n_jobs)
                
        study.optimize(lambda trial: self.hp_search(trial, train_data, val_data, self.DataConfig, self.tune_dir), 
                        n_trials=num_trials, 
                        n_jobs=max_workers)
        
        with open(self.tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = study.best_trial.number), "rb") as fin:
            self.best_model = pickle.load(fin)
        
    def hp_search(self, 
                   trial: optuna.Trial,
                   train_data:dict,
                   val_data:dict,
                   DataConfig:dict,
                   tune_dir:str,
                   retrain_with_valid:bool= True,
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
        
        hyper_params = {
                        'eta': trial.suggest_float('eta ', DataConfig['hyper_params']['eta']['min'], DataConfig['hyper_params']['eta']['max']),
                        'gamma': trial.suggest_float('gamma ', DataConfig['hyper_params']['gamma']['min'], DataConfig['hyper_params']['gamma']['max']),
                        'max_depth': trial.suggest_int('max_depth', DataConfig['hyper_params']['max_depth']['min'], DataConfig['hyper_params']['max_depth']['max'], step=DataConfig['hyper_params']['max_depth']['step']),
                        'min_child_weight': trial.suggest_int('min_child_weight', DataConfig['hyper_params']['min_child_weight']['min'], DataConfig['hyper_params']['min_child_weight']['max'], step=DataConfig['hyper_params']['min_child_weight']['step']),
                        'subsample': trial.suggest_float('subsample ', DataConfig['hyper_params']['subsample']['min'], DataConfig['hyper_params']['subsample']['max'], step=DataConfig['hyper_params']['subsample']['step']),
                        'colsample_bynode': trial.suggest_float('colsample_bynode ', DataConfig['hyper_params']['colsample_bynode']['min'], DataConfig['hyper_params']['colsample_bynode']['max'], step=DataConfig['hyper_params']['colsample_bynode']['step']),
                        'lambda': trial.suggest_float('lambda ', DataConfig['hyper_params']['lambda']['min'], DataConfig['hyper_params']['lambda']['max']),
                        'alpha': trial.suggest_float('alpha ', DataConfig['hyper_params']['alpha']['min'], DataConfig['hyper_params']['alpha']['max']),
                        'tree_method': trial.suggest_categorical('tree_method', DataConfig['hyper_params']['tree_method']),
                        'n_jobs': 1,
                        'num_parallel_tree':1,
                        'random_state':None}
        
        training_params = {'num_boost_round': trial.suggest_int('num_boost_round', DataConfig['hyper_params']['num_boost_round']['min'], DataConfig['hyper_params']['num_boost_round']['max'], step=DataConfig['hyper_params']['num_boost_round']['step'])}
        training_params['early_stopping_rounds'] = 10
        
        if self.method == "XGBoostRegressor":
            hyper_params['objective'] = "reg:pseudohubererror"
            hyper_params['eval_metric'] = "mphe"      
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-mphe")

        elif self.method == "XGBoostClassifier":
            hyper_params['objective'] = "binary:logistic"
            hyper_params['eval_metric'] = "logloss"
            hyper_params['scale_pos_weight'] = len(train_data['target'][train_data['target']]==0) / len(train_data['target'][train_data['target']]==1)
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-logloss")

        dtrain = xgb.DMatrix(train_data['features'], label=train_data['target'])
        
        if oversampling and self.method == "XGBoostRegressor":
            
            age_classes, current_points = np.unique(np.round(train_data['target'], -1), return_counts=True)
            desired_points_per_class = np.nanmax(current_points)
            
            Y_sample = []
            X_sample = []
            for a, b in zip(age_classes, current_points):
                
                required_samples = desired_points_per_class - b
                
                if required_samples > 0:
                    if self.method == "XGBoostRegressor":
                        idx_ =  np.where(np.round(train_data['target'], -1) == a)[0]   
                    
                    elif self.method == "XGBoostClassifier":
                        idx_ =  np.where(train_data['target'] == a)[0]   
                        
                    idx_sample = np.random.choice(idx_, required_samples)
                    Y_sample.append(train_data['target'][idx_sample]), 
                    X_sample.append(train_data['features'][idx_sample])
                    
            Y_sample = np.concatenate(Y_sample)
            X_sample = np.concatenate(X_sample)
            dtrain = xgb.DMatrix(np.concatenate([train_data['features'], X_sample]), 
                                  label=np.concatenate([train_data['target'], Y_sample]))

        deval = xgb.DMatrix(val_data['features'], label = val_data['target'])
        vallist = [(dtrain, 'train'), (deval, 'eval')]
        
        first_model = xgb.train(hyper_params, dtrain, evals=vallist, callbacks = [pruning_callback],
                           verbose_eval=False, **training_params)

        if retrain_with_valid:
            training_params['num_boost_round'] = first_model.best_ntree_limit
            training_params['early_stopping_rounds'] = None 
            
            if oversampling and self.method == "XGBoostRegressor":
                
                age_classes, current_points = np.unique(np.round(np.concatenate([train_data['target'], val_data['target']]), -1), return_counts=True)
                desired_points_per_class = np.nanmax(current_points)
                
                Y_sample = []
                X_sample = []
                for a, b in zip(age_classes, current_points):
                    
                    required_samples = desired_points_per_class - b
                    
                    if required_samples > 0:
                        if self.method == "XGBoostRegressor":
                            idx_ =  np.where(np.round(np.concatenate([train_data['target'], val_data['target']]), -1) == a)[0]   
                        
                        elif self.method == "XGBoostClassifier":
                            idx_ =  np.where(np.concatenate([train_data['target'], val_data['target']]) == a)[0]   
                        
                        idx_sample = np.random.choice(idx_, required_samples)
                        Y_sample.append(np.concatenate([train_data['target'], val_data['target']])[idx_sample]), 
                        X_sample.append(np.concatenate([train_data['features'], val_data['features']])[idx_sample])
                        
                Y_sample = np.concatenate(Y_sample)
                X_sample = np.concatenate(X_sample)
                dtrain = xgb.DMatrix(np.concatenate([train_data['features'], val_data['features'], X_sample]), 
                                     label = np.concatenate([train_data['target'], val_data['target'], Y_sample]))
            else:
                dtrain = xgb.DMatrix(np.concatenate([train_data['features'], val_data['features']]), 
                                     label = np.concatenate([train_data['target'], val_data['target']]))
                 
            model_ = xgb.train(hyper_params, dtrain, **training_params) 
        
        else: 
            model_= first_model            
            
        with open(tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        return first_model.best_score
    
    def predict_clusters(self, 
                        save_cube:str) -> None:
        """Predict the target variables for the given data and save the results in a prediction zarr cube.
    
        Parameters:
            save_cube: str
                Path to the output netCDF file where the predictions will be saved.
        """
        
        X = self.mldata.test_dataloader().get_x(method= self.method, features = self.DataConfig['features_selected'])
        Y = self.mldata.test_dataloader().get_y(target= self.DataConfig['target'], 
                                               method= self.method, 
                                               max_forest_age= self.DataConfig['max_forest_age'])

        for cluster_ in np.arange(len(self.mldata.test_subset)):
            X_cluster = X[cluster_, : , :].reshape(-1, len(self.DataConfig['features_selected']))
            Y_cluster = Y[:, cluster_, :].reshape(-1)
            mask_nan = (np.all(np.isfinite(X_cluster), axis=1)) & (np.isfinite(Y_cluster))
            if X_cluster[mask_nan, :].shape[0]>0:
                dpred =  xgb.DMatrix(X_cluster[mask_nan, :])
                if self.method == "XGBoostRegressor":
                    y_hat =  self.best_model.predict(dpred)
                    
                elif self.method == "XGBoostClassifier":
                    #y_hat =  np.rint(self.best_model.predict(dpred))
                    y_hat_proba =  self.best_model.predict(dpred)                    
                    y_hat = (y_hat_proba > 0.5).astype(int)                  
                    
                preds = xr.Dataset()
                
                if self.method == "XGBoostClassifier": 
                    out_var = 'oldGrowth'
                elif self.method == "XGBoostRegressor": 
                    out_var = 'forestAge'
                
                preds["{out_var}_pred".format(out_var = out_var)] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["{out_var}_obs".format(out_var = out_var)] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
        
                if self.method == "XGBoostClassifier": 
                    preds["{out_var}_proba".format(out_var = out_var)] = xr.DataArray([y_hat_proba], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat_proba))})
                    
                save_cube.CubeWriter(preds.transpose('sample', 'cluster'))
   
    
    