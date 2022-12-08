#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   MLP.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for training MLP model
"""
#%% Load library
import os
from typing import Any
import numpy as np
import pickle
import optuna
from sklearn.metrics import mean_squared_error, log_loss
import xarray as xr
from sklearn.neural_network import MLPRegressor, MLPClassifier
from ageUpscaling.dataloaders.ml_dataloader import MLDataModule
from ageUpscaling.methods.feature_selection import FeatureSelection

class MLPmethod:
    """MLPMethod(tune_dir:str=None)
    A method class for training MLP model
    
    Parameters
    ----------
    tune_dir : str, default is None
    
        string defining the directory to save the model experiment
        
    Return
    ------
    a model: the best model of the optuna experiment
    """
    
    def __init__(
            self,
            tune_dir: str=None,
            DataConfig:dict=None) -> None:

        self.tune_dir = tune_dir
        
        if not os.path.exists(tune_dir):
            os.makedirs(tune_dir)
            
        self.DataConfig = DataConfig    
        
    def get_datamodule(
            self, 
            DataConfig: dict[str, Any] = {},
            target: dict[str, Any] = {},
            features: dict[str, Any] = {},
            train_subset: dict[str, Any] = {},
            valid_subset: dict[str, Any] = {},
            test_subset: dict[str, Any] = {},
            **kwargs) -> dict:
        
        mlData = MLDataModule(DataConfig,
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

        if feature_selection:
            self.mldata = self.get_datamodule(DataConfig=self.DataConfig, 
                                              target=self.DataConfig['target'],
                                              features =  self.DataConfig['features'],
                                              train_subset=train_subset,
                                              valid_subset=valid_subset,
                                              test_subset=test_subset)            
            train_data = self.mldata.train_dataloader().get_xy()
            features_selected = FeatureSelection(method=self.DataConfig['method'][0], 
                                                 feature_selection_method = feature_selection_method, 
                                                 features = self.DataConfig['features']).get_features(data = train_data)
        
        self.final_features = [features_selected if feature_selection else self.DataConfig['features']][0]
        
        self.mldata = self.get_datamodule(DataConfig=self.DataConfig, 
                                          target=self.DataConfig['target'],
                                          features = self.final_features,
                                          train_subset=train_subset,
                                          valid_subset=valid_subset,
                                          test_subset=test_subset)
          
        train_data = self.mldata.train_dataloader().get_xy()
        val_data = self.mldata.val_dataloader().get_xy()
                    
        if not os.path.exists(self.tune_dir + '/trial_model/'):
            os.makedirs(self.tune_dir + '/trial_model/')
        
        study = optuna.create_study(study_name = 'hpo_ForestAge', 
                                    storage='sqlite:///' + self.tune_dir + '/trial_model/hp_trial.db',
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                   reduction_factor=4, 
                                                                                   min_early_stopping_rate=8),
                                    direction='minimize')
        study.optimize(lambda trial: self.hp_search(trial, train_data, val_data, self.DataConfig, self.tune_dir), 
                       n_trials=self.DataConfig['hyper_params']['number_trials'], n_jobs=n_jobs)
        
        with open(self.tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = study.best_trial.number), "rb") as fin:
            self.best_model = pickle.load(fin)            
            
    def hp_search(self, 
                   trial: optuna.Trial,
                   train_data:dict,
                   val_data:dict,
                   DataConfig:dict,
                   tune_dir:str) -> float:
        
        hyper_params = {
            'learning_rate_init': trial.suggest_float('learning_rate_init ', DataConfig['hyper_params']['learning_rate_init']['min'], DataConfig['hyper_params']['learning_rate_init']['max'], step=DataConfig['hyper_params']['learning_rate_init']['step']),
            'learning_rate': trial.suggest_categorical('learning_rate', DataConfig['hyper_params']['learning_rate']),
            'first_layer_neurons': trial.suggest_int('first_layer_neurons', DataConfig['hyper_params']['first_layer_neurons']['min'], DataConfig['hyper_params']['first_layer_neurons']['max'], step=DataConfig['hyper_params']['first_layer_neurons']['step']),
            'second_layer_neurons': trial.suggest_int('second_layer_neurons', DataConfig['hyper_params']['second_layer_neurons']['min'], DataConfig['hyper_params']['second_layer_neurons']['max'], step=DataConfig['hyper_params']['second_layer_neurons']['step']),
            'third_layer_neurons': trial.suggest_int('third_layer_neurons', DataConfig['hyper_params']['third_layer_neurons']['min'], DataConfig['hyper_params']['third_layer_neurons']['max'], step=DataConfig['hyper_params']['third_layer_neurons']['step']),
            'activation': trial.suggest_categorical('activation', DataConfig['hyper_params']['activation']),
            'solver': trial.suggest_categorical('solver', DataConfig['hyper_params']['solver']),            
            'batch_size': trial.suggest_int('batch_size', DataConfig['hyper_params']['batch_size']['min'], DataConfig['hyper_params']['batch_size']['max'], step=DataConfig['hyper_params']['batch_size']['step'])}
        
        if self.DataConfig['method'][0] == "MLPRegressor": 
            model_ = MLPRegressor(
                        hidden_layer_sizes=(hyper_params['first_layer_neurons'], 
                                            hyper_params['second_layer_neurons'],
                                            hyper_params['third_layer_neurons'],
                                            ),
                       learning_rate_init=hyper_params['learning_rate_init'],
                       learning_rate = hyper_params['learning_rate'],
                       activation=hyper_params['activation'],
                       solver = hyper_params['solver'],
                       batch_size=hyper_params['batch_size'],
                       random_state=1)
        elif self.DataConfig['method'][0] == "MLPClassifier": 
            model_ = MLPClassifier(
                        hidden_layer_sizes=(hyper_params['first_layer_neurons'], 
                                           hyper_params['second_layer_neurons']),
                       learning_rate_init=hyper_params['learning_rate_init'],
                       learning_rate = hyper_params['learning_rate'],
                       activation=hyper_params['activation'],
                       solver = hyper_params['solver'],
                       batch_size=hyper_params['batch_size'],
                       random_state=1)
        model_.fit(train_data['features'], train_data['target'])
        
        with open(tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if self.DataConfig['method'][0] == "MLPRegressor":
            loss_ = mean_squared_error(val_data['target'], model_.predict(val_data['features']), squared=False)
        if self.DataConfig['method'][0] == "MLPClassifier":     
            loss_ =  log_loss(val_data['target'], model_.predict(val_data['features']))
        
        return loss_
    
    def predict_clusters(
            self, 
            save_cube:str) -> xr.Dataset:
        
        X = self.mldata.test_dataloader().get_x(features= self.final_features)
        Y = self.mldata.test_dataloader().get_y(target= self.DataConfig['target'], 
                                               method= self.DataConfig['method'][0], 
                                               max_forest_age= self.DataConfig['max_forest_age'])

        for cluster_ in np.arange(len(self.mldata.test_subset)):
            X_cluster = X[cluster_, : , :].reshape(-1, len(self.final_features))
            Y_cluster = Y[:, cluster_, :].reshape(-1)
            mask_nan = np.isfinite(Y_cluster)
            if X_cluster[mask_nan, :].shape[0]>0:
                y_hat = self.best_model.predict(X_cluster[mask_nan, :])
                preds = xr.Dataset()
                preds["forestAge_pred"] = xr.DataArray([self.denorm_target(y_hat)], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["forestAge_obs"] = xr.DataArray([self.denorm_target(Y_cluster[mask_nan])], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                save_cube.compute_cube(preds, initialize=True, njobs=1)
                
    def denorm_target(self, 
             x: np.array) -> np.array:
        """Returns de-normalize target, last dimension of `x` must match len of `self.target_norm_stats`."""
        
        return x * self.mldata.norm_stats[self.DataConfig['target'][0]]['std'] + self.mldata.norm_stats[self.DataConfig['target'][0]]['mean']
                

    