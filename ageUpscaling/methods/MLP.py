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
              mlp_method:str = 'MLPRegressor', 
              train_subset:dict={},
              valid_subset:dict={}) -> None:

        mldata = self.get_datamodule(cube_path=cube_path,
                                     data_config=self.data_config,
                                     train_subset=train_subset,
                                     valid_subset=valid_subset)

        if not os.path.exists(self.save_dir + '/save_model/{method}/'.format(method = mlp_method)):
            os.makedirs(self.save_dir + '/save_model/{method}/'.format(method = mlp_method))
        
        study = optuna.create_study(study_name = 'hpo_ForestAge', 
                                    storage='sqlite:///' + self.save_dir + '/save_model/{method}//hp_trial.db'.format(method = mlp_method),
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                   reduction_factor=4, 
                                                                                   min_early_stopping_rate=0),
                                    direction='minimize')
        study.optimize(lambda trial: self.hp_search(trial, mlp_method, self.data_config, mldata, self.save_dir), 
                       n_trials=300, n_jobs=4)
        
        with open(self.save_dir + "/save_model/model_trial_{}.pickle".format(study.best_trial.number), "rb") as fin:
            self.best_model = pickle.load(fin)            
            
    def hp_search(self, 
                   trial: optuna.Trial,
                   mlp_method:str,
                   data_config:dict,
                   mldata:dict,
                   save_dir:str) -> float:
        
        hyper_params = {
            'learning_rate_init': trial.suggest_float('learning_rate_init ', data_config['hyper_params']['learning_rate_init']['min'], data_config['hyper_params']['learning_rate_init']['max'], step=data_config['hyper_params']['learning_rate_init']['step']),
            'first_layer_neurons': trial.suggest_int('first_layer_neurons', data_config['hyper_params']['first_layer_neurons']['min'], data_config['hyper_params']['first_layer_neurons']['max'], step=data_config['hyper_params']['first_layer_neurons']['step']),
            'second_layer_neurons': trial.suggest_int('second_layer_neurons', data_config['hyper_params']['second_layer_neurons']['min'], data_config['hyper_params']['second_layer_neurons']['max'], step=data_config['hyper_params']['second_layer_neurons']['step']),
            'activation': trial.suggest_categorical('activation', data_config['hyper_params']['activation']),
            'batch_size': trial.suggest_int('batch_size', data_config['hyper_params']['batch_size']['min'], data_config['hyper_params']['batch_size']['max'], step=data_config['hyper_params']['batch_size']['step'])}
        
        if mlp_method == "MLPRegressor": 
            model_ = MLPRegressor(
                        hidden_layer_sizes=(hyper_params['first_layer_neurons'], 
                                           hyper_params['second_layer_neurons']),
                       learning_rate_init=hyper_params['learning_rate_init'],
                       activation=hyper_params['activation'],
                       batch_size=hyper_params['batch_size'],
                       random_state=1,
                       max_iter=100, 
                       early_stopping= True, 
                       validation_fraction = 0.3)
        elif mlp_method == "MLPClassifier": 
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
        
        model_.fit(mldata.train_dataloader().get_xy()['features'], mldata.train_dataloader().get_xy()['target'])
        
        with open(save_dir + '/save_model/model_trial{}.pickle'.format(trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if mlp_method == "MLPRegressor":
            loss_ = mean_squared_error(mldata.val_dataloader().get_xy()['target'], model_.predict(mldata.val_dataloader().get_xy()['features']), squared=False)
        if mlp_method == "MLPClassifier":            
            loss_ =  log_loss(mldata.val_dataloader().get_xy()['target'], model_.predict(mldata.val_dataloader().get_xy()['features']))
        
        return loss_
    
    def predict(self,
                mldata:dict) -> np.array:
            
        pred_ = self.best_model.predict(mldata.test_dataloader().get_xy()['features'])
                
        return pred_
    
    def predict_xr(
            self,
            x: xr.Dataset,
            data_config: dict, 
            device=None) -> xr.Dataset:

        if 'sample' not in x.dim:
            raise ValueError(
                '`x` must have dimension \'sample\'.'
            )

        if 'time' not in x.dim:
            raise ValueError(
                '`x` must have dimension \'time\'.'
            )

        x, s = MLDataModule.x2mod(x=x, norm_stats=self.norm_stats)
      
        y_hat_norm = self.model(x)
        y_hat = MLDataModule.denorm(y_hat_norm, norm_stats=self.norm_stats)

        preds = xr.Dataset()
        for var in data_config.target:
            preds[var.name] = xr.DataArray(y_hat, coords=[x.sample, x.time])

        return preds


    