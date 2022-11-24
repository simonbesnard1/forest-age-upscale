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
            train_subset: dict[str, Any] = {},
            valid_subset: dict[str, Any] = {},
            test_subset: dict[str, Any] = {},
            **kwargs) -> dict:
        
        mlData = MLDataModule(DataConfig,
                              train_subset, 
                              valid_subset, 
                              test_subset)

        return mlData
        
    def train(self,  
              train_subset:dict={},
              valid_subset:dict={}, 
              test_subset:dict={},
              feature_selection:bool= False) -> None:

        self.mldata = self.get_datamodule(DataConfig=self.DataConfig,
                                         train_subset=train_subset,
                                         valid_subset=valid_subset,
                                         test_subset=test_subset)
        
        train_data = self.mldata.train_dataloader().get_xy()
        val_data = self.mldata.val_dataloader().get_xy()
        if feature_selection:
            feature_method = FeatureSelection(model='regression', selection_method = "boruta")
            feature_selected = feature_method.get_features(data = train_data)
            
        if not os.path.exists(self.tune_dir + '/save_model/'):
            os.makedirs(self.tune_dir + '/save_model/')
        
        study = optuna.create_study(study_name = 'hpo_ForestAge', 
                                    storage='sqlite:///' + self.tune_dir + '/save_model/hp_trial.db',
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                   reduction_factor=4, 
                                                                                   min_early_stopping_rate=0),
                                    direction='minimize')
        study.optimize(lambda trial: self.hp_search(trial, train_data, val_data, self.DataConfig, self.tune_dir), 
                       n_trials=1, n_jobs=4)
        
        with open(self.tune_dir + "/save_model/model_trial_{id_}.pickle".format(id_ = study.best_trial.number), "rb") as fin:
            self.best_model = pickle.load(fin)            
            
    def hp_search(self, 
                   trial: optuna.Trial,
                   train_data:dict,
                   val_data:dict,
                   DataConfig:dict,
                   tune_dir:str) -> float:
        
        hyper_params = {
            'learning_rate_init': trial.suggest_float('learning_rate_init ', DataConfig['hyper_params']['learning_rate_init']['min'], DataConfig['hyper_params']['learning_rate_init']['max'], step=DataConfig['hyper_params']['learning_rate_init']['step']),
            'first_layer_neurons': trial.suggest_int('first_layer_neurons', DataConfig['hyper_params']['first_layer_neurons']['min'], DataConfig['hyper_params']['first_layer_neurons']['max'], step=DataConfig['hyper_params']['first_layer_neurons']['step']),
            'second_layer_neurons': trial.suggest_int('second_layer_neurons', DataConfig['hyper_params']['second_layer_neurons']['min'], DataConfig['hyper_params']['second_layer_neurons']['max'], step=DataConfig['hyper_params']['second_layer_neurons']['step']),
            'third_layer_neurons': trial.suggest_int('third_layer_neurons', DataConfig['hyper_params']['third_layer_neurons']['min'], DataConfig['hyper_params']['third_layer_neurons']['max'], step=DataConfig['hyper_params']['third_layer_neurons']['step']),
            'activation': trial.suggest_categorical('activation', DataConfig['hyper_params']['activation']),
            'batch_size': trial.suggest_int('batch_size', DataConfig['hyper_params']['batch_size']['min'], DataConfig['hyper_params']['batch_size']['max'], step=DataConfig['hyper_params']['batch_size']['step'])}
        
        if self.DataConfig['method'][0] == "MLPRegressor": 
            model_ = MLPRegressor(
                        hidden_layer_sizes=(hyper_params['first_layer_neurons'], 
                                            hyper_params['second_layer_neurons'],
                                            hyper_params['third_layer_neurons'],
                                            ),
                       learning_rate_init=hyper_params['learning_rate_init'],
                       activation=hyper_params['activation'],
                       batch_size=hyper_params['batch_size'],
                       random_state=1,
                       max_iter=100, 
                       early_stopping= True, 
                       validation_fraction = 0.3)
        elif self.DataConfig['method'][0] == "MLPClassifier": 
            model_ = MLPClassifier(
                        hidden_layer_sizes=(hyper_params['first_layer_neurons'], 
                                           hyper_params['second_layer_neurons']),
                       learning_rate_init=hyper_params['learning_rate_init'],
                       activation=hyper_params['activation'],
                       batch_size=hyper_params['batch_size'],
                       random_state=1,
                       max_iter=100, 
                       early_stopping= True, 
                       validation_fraction = 0.3)
        model_.fit(train_data['features'], train_data['target'])
        
        with open(tune_dir + "/save_model/model_trial_{id_}.pickle".format(id_ = trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if self.DataConfig['method'][0] == "MLPRegressor":
            loss_ = mean_squared_error(val_data['target'], model_.predict(val_data['features']), squared=False)
        if self.DataConfig['method'][0] == "MLPClassifier":            
            loss_ =  log_loss(val_data['target'], model_.predict(val_data['features']))
        
        return loss_
    
    def predict_xr(
            self, 
            save_cube:str) -> xr.Dataset:
        
        X = self.mldata.test_dataloader().get_x(features= self.DataConfig['features'])
        Y = self.mldata.test_dataloader().get_y(target= self.DataConfig['target'], 
                                                method= self.DataConfig['method'][0], 
                                                max_forest_age= self.DataConfig['max_forest_age'])
        
        for cluster_ in np.arange(len(self.mldata.test_subset)):
            X_cluster = X[cluster_, : , :].reshape(-1, len(self.DataConfig['features']))
            Y_cluster = Y[:, cluster_, :].reshape(-1)
            mask_nan = np.isfinite(Y_cluster)
            try:
                y_hat = self.best_model.predict(X_cluster[mask_nan, :])
                preds = xr.Dataset()
                preds["forestAge_pred"] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["forestAge_obs"] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                save_cube.update_cube(preds, initialize=True, is_sorted=False, njobs=1)
            except:
                print('cluster_{id_} has only NaN values'.format(id_ = str(self.mldata.test_subset[cluster_])))


    