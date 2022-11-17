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
import yaml as yml
from sklearn.neural_network import MLPRegressor, MLPClassifier
from ageUpscaling.dataloaders.ml_dataloader import MLDataModule

class MLPmethod:
    """MLPMethod(save_dir:str=None)
    A method class for training MLP model
    
    Parameters
    ----------
    save_dir : str, default is None
    
        string defining the directory to save the model experiment
        
    Return
    ------
    a model: the best model of the optuna experiment
    """
    
    def __init__(
            self,
            save_dir: str=None,
            data_config_path:str=None) -> None:

        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        with open(data_config_path, 'r') as f:
            self.data_config =  yml.safe_load(f)
        
    def get_datamodule(
            self,
            cube_path:str, 
            data_config: dict[str, Any] = {},
            train_subset: dict[str, Any] = {},
            valid_subset: dict[str, Any] = {},
            test_subset: dict[str, Any] = {},
            **kwargs) -> dict:
        
        mlData = MLDataModule(cube_path, 
                              data_config,
                              train_subset, 
                              valid_subset, 
                              test_subset)

        return mlData
        
    def train(self, 
              cube_path:np.array = [], 
              train_subset:dict={},
              valid_subset:dict={}, 
              test_subset:dict={}) -> None:

        self.mldata = self.get_datamodule(cube_path=cube_path,
                                         data_config=self.data_config,
                                         train_subset=train_subset,
                                         valid_subset=valid_subset,
                                         test_subset=test_subset)
        
        train_data = self.mldata.train_dataloader().get_xy()
        val_data = self.mldata.val_dataloader().get_xy()
        
        if not os.path.exists(self.save_dir + '/save_model/{method}/'.format(method = self.data_config['method'][0])):
            os.makedirs(self.save_dir + '/save_model/{method}/'.format(method = self.data_config['method'][0]))
        
        study = optuna.create_study(study_name = 'hpo_ForestAge', 
                                    storage='sqlite:///' + self.save_dir + '/save_model/{method}/hp_trial.db'.format(method = self.data_config['method'][0]),
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                   reduction_factor=4, 
                                                                                   min_early_stopping_rate=0),
                                    direction='minimize')
        study.optimize(lambda trial: self.hp_search(trial, train_data, val_data, self.data_config, self.save_dir), 
                       n_trials=2, n_jobs=4)
        
        with open(self.save_dir + "/save_model/{method}/model_trial_{id_}.pickle".format(method = self.data_config['method'][0], id_ = study.best_trial.number), "rb") as fin:
            self.best_model = pickle.load(fin)            
            
    def hp_search(self, 
                   trial: optuna.Trial,
                   train_data:dict,
                   val_data:dict,
                   data_config:dict,
                   save_dir:str) -> float:
        
        hyper_params = {
            'learning_rate_init': trial.suggest_float('learning_rate_init ', data_config['hyper_params']['learning_rate_init']['min'], data_config['hyper_params']['learning_rate_init']['max'], step=data_config['hyper_params']['learning_rate_init']['step']),
            'first_layer_neurons': trial.suggest_int('first_layer_neurons', data_config['hyper_params']['first_layer_neurons']['min'], data_config['hyper_params']['first_layer_neurons']['max'], step=data_config['hyper_params']['first_layer_neurons']['step']),
            'second_layer_neurons': trial.suggest_int('second_layer_neurons', data_config['hyper_params']['second_layer_neurons']['min'], data_config['hyper_params']['second_layer_neurons']['max'], step=data_config['hyper_params']['second_layer_neurons']['step']),
            'third_layer_neurons': trial.suggest_int('third_layer_neurons', data_config['hyper_params']['third_layer_neurons']['min'], data_config['hyper_params']['third_layer_neurons']['max'], step=data_config['hyper_params']['third_layer_neurons']['step']),
            'activation': trial.suggest_categorical('activation', data_config['hyper_params']['activation']),
            'batch_size': trial.suggest_int('batch_size', data_config['hyper_params']['batch_size']['min'], data_config['hyper_params']['batch_size']['max'], step=data_config['hyper_params']['batch_size']['step'])}
        
        if self.data_config['method'][0] == "MLPRegressor": 
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
        elif self.data_config['method'][0] == "MLPClassifier": 
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
        
        with open(save_dir + "/save_model/{method}/model_trial_{id_}.pickle".format(method = self.data_config['method'][0], id_ = trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if self.data_config['method'][0] == "MLPRegressor":
            loss_ = mean_squared_error(val_data['target'], model_.predict(val_data['features']), squared=False)
        if self.data_config['method'][0] == "MLPClassifier":            
            loss_ =  log_loss(val_data['target'], model_.predict(val_data['features']))
        
        return loss_
    
    def predict_xr(
            self, 
            save_cube:str) -> xr.Dataset:
        
        X = self.mldata.test_dataloader().get_x(features= self.data_config['features'])
        Y = self.mldata.test_dataloader().get_y(target= self.data_config['target'], 
                                                method= self.data_config['method'][0], 
                                                max_forest_age= self.data_config['max_forest_age'])
        
        for cluster_ in np.arange(len(self.mldata.test_subset)):
            X_cluster = X[cluster_, : , :].reshape(-1, len(self.data_config['features']))
            Y_cluster = Y[:, cluster_, :].reshape(-1)
            mask_nan = np.isfinite(Y_cluster)
            y_hat = self.best_model.predict(X_cluster[mask_nan, :])
            preds = xr.Dataset()
            preds["forestAge_pred"] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
            preds["forestAge_obs"] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
       
        return preds


    