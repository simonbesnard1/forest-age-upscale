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
import os
import numpy as np
import pickle
from typing import Any

import xarray as xr

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score

import optuna

from ageUpscaling.dataloaders.ml_dataloader import MLDataModule
from ageUpscaling.methods.feature_selection import FeatureSelection

class MLPmethod:
    """A method class for training and evaluating an MLP model.
    
    Parameters
    ----------
    tune_dir : str, default is None
        Directory to save the model experiment. If not provided, the model experiment will not be saved.
    DataConfig : dict, default is None
        Dictionary containing the data configuration.
    method : str, default is 'MLPRegressor'
        String defining the type of MLP model to use. Can be 'MLPRegressor' for a regression model or 'MLPClassifier' for a classification model.
    """
    
    def __init__(self,
                 tune_dir: str=None,
                 DataConfig:dict=None,
                 method:str = 'MLPRegressor') -> None:

        self.tune_dir = tune_dir
        
        if not os.path.exists(tune_dir):
            os.makedirs(tune_dir)
            
        self.DataConfig = DataConfig
        self.method = method
        
    def get_datamodule(self, 
                       method:str = 'MLPRegressor',
                       DataConfig: dict[str, Any] = {},
                       target: dict[str, Any] = {},
                       features: dict[str, Any] = {},
                       train_subset: dict[str, Any] = {},
                       valid_subset: dict[str, Any] = {},
                       test_subset: dict[str, Any] = {},
                       normalize:bool= False,
                       **kwargs) -> MLDataModule:
        """Returns the data module for training the model.

        Parameters:
            method: str, default is 'MLPRegressor'
                The type of model to use for training ('MLPRegressor' or 'MLPClassifier').
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
                              test_subset,
                              normalize)

        return mlData
        
    def train(self,  
              train_subset:dict={},
              valid_subset:dict={}, 
              test_subset:dict={},
              feature_selection:bool= False,
              feature_selection_method:str="recursive", 
              n_jobs:int=10) -> None:
        
        """Trains an MLP model using the specified training and validation datasets.

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
                                          features = self.final_features,
                                          train_subset=train_subset,
                                          valid_subset=valid_subset,
                                          test_subset=test_subset,
                                          normalize= True)
          
        train_data = self.mldata.train_dataloader().get_xy()
        val_data = self.mldata.val_dataloader().get_xy()
                    
        if not os.path.exists(self.tune_dir + '/trial_model/'):
            os.makedirs(self.tune_dir + '/trial_model/')
        
        study = optuna.create_study(study_name = 'hpo_ForestAge', 
                                    #storage='sqlite:///' + self.tune_dir + '/trial_model/hp_trial.db',
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                   reduction_factor=4, 
                                                                                   min_early_stopping_rate=8),
                                    direction=['minimize' if self.method == 'MLPRegressor' else 'maximize'][0])
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
            'learning_rate_init': trial.suggest_float('learning_rate_init ', DataConfig['hyper_params']['learning_rate_init']['min'], DataConfig['hyper_params']['learning_rate_init']['max'], step=DataConfig['hyper_params']['learning_rate_init']['step']),
            'learning_rate': trial.suggest_categorical('learning_rate', DataConfig['hyper_params']['learning_rate']),
            'num_layers': trial.suggest_int('num_layers', DataConfig['hyper_params']['num_layers']['min'], DataConfig['hyper_params']['num_layers']['max'], step=DataConfig['hyper_params']['num_layers']['step']),
            'first_layer_neurons': trial.suggest_int('first_layer_neurons', DataConfig['hyper_params']['first_layer_neurons']['min'], DataConfig['hyper_params']['first_layer_neurons']['max'], step=DataConfig['hyper_params']['first_layer_neurons']['step']),
            'second_layer_neurons': trial.suggest_int('second_layer_neurons', DataConfig['hyper_params']['second_layer_neurons']['min'], DataConfig['hyper_params']['second_layer_neurons']['max'], step=DataConfig['hyper_params']['second_layer_neurons']['step']),
            'third_layer_neurons': trial.suggest_int('third_layer_neurons', DataConfig['hyper_params']['third_layer_neurons']['min'], DataConfig['hyper_params']['third_layer_neurons']['max'], step=DataConfig['hyper_params']['third_layer_neurons']['step']),
            'fourth_layer_neurons': trial.suggest_int('fourth_layer_neurons', DataConfig['hyper_params']['fourth_layer_neurons']['min'], DataConfig['hyper_params']['fourth_layer_neurons']['max'], step=DataConfig['hyper_params']['fourth_layer_neurons']['step']),
            'activation': trial.suggest_categorical('activation', DataConfig['hyper_params']['activation']),
            'solver': trial.suggest_categorical('solver', DataConfig['hyper_params']['solver']),            
            'batch_size': trial.suggest_int('batch_size', DataConfig['hyper_params']['batch_size']['min'], DataConfig['hyper_params']['batch_size']['max'], step=DataConfig['hyper_params']['batch_size']['step']),
            'tol': trial.suggest_float('tol ', DataConfig['hyper_params']['tol']['min'], DataConfig['hyper_params']['tol']['max'], step=DataConfig['hyper_params']['tol']['step']),
            
            }
        
        if self.method == "MLPRegressor": 
            model_ = MLPRegressor(hidden_layer_sizes=tuple([hyper_params['first_layer_neurons'], 
                                                            hyper_params['second_layer_neurons'],
                                                            hyper_params['third_layer_neurons'],
                                                            hyper_params['fourth_layer_neurons']][0:hyper_params['num_layers']]),
                                   learning_rate_init=hyper_params['learning_rate_init'],
                                   learning_rate = hyper_params['learning_rate'],
                                   activation=hyper_params['activation'],
                                   solver = hyper_params['solver'],
                                   batch_size=hyper_params['batch_size'],
                                   warm_start=True,
                                   max_iter=200, 
                                   tol=hyper_params['tol'],
                                   n_iter_no_change= 10,
                                   #early_stopping= True, 
                                   #validation_fraction=0.3,
                                   random_state=1)
            
        elif self.method == "MLPClassifier": 
            model_ = MLPClassifier(hidden_layer_sizes=tuple([hyper_params['first_layer_neurons'], 
                                                            hyper_params['second_layer_neurons'],
                                                            hyper_params['third_layer_neurons'],
                                                            hyper_params['fourth_layer_neurons']][0:hyper_params['num_layers']]),
                                   learning_rate_init=hyper_params['learning_rate_init'],
                                   learning_rate = hyper_params['learning_rate'],
                                   activation=hyper_params['activation'],
                                   solver = hyper_params['solver'],
                                   batch_size=hyper_params['batch_size'],
                                   warm_start=True,
                                   max_iter=200, 
                                   tol=hyper_params['tol'],
                                   n_iter_no_change= 10,
                                   #early_stopping= True, 
                                   #validation_fraction=0.3,
                                   random_state=1)
        
        model_.fit(train_data['features'], train_data['target'])
        
        with open(tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if self.method == "MLPRegressor":
            loss_ = mean_squared_error(val_data['target'], model_.predict(val_data['features']), squared=False)
        elif self.method == "MLPClassifier":
            loss_ =  roc_auc_score(val_data['target'], model_.predict(val_data['features']))
        return loss_ 
    
    def predict_clusters(self, 
                        save_cube:str) -> None:
        """Predict the target variables for the given data and save the results in a prediction zarr cube.
    
        Parameters:
            save_cube: str
                Path to the output netCDF file where the predictions will be saved.
        """
        
        X = self.mldata.test_dataloader().get_x(method= self.method, features= self.final_features)
        Y = self.mldata.test_dataloader().get_y(target= self.DataConfig['target'], 
                                               method= self.method, 
                                               max_forest_age= self.DataConfig['max_forest_age'])

        for cluster_ in np.arange(len(self.mldata.test_subset)):
            X_cluster = X[cluster_, : , :].reshape(-1, len(self.final_features))
            Y_cluster = Y[:, cluster_, :].reshape(-1)
            mask_nan = (np.all(np.isfinite(X_cluster), axis=1)) & (np.isfinite(Y_cluster))
            if X_cluster[mask_nan, :].shape[0]>0:
                y_hat = self.best_model.predict(X_cluster[mask_nan, :])
                preds = xr.Dataset()
                
                if self.method == "MLPClassifier": 
                    out_var = 'oldGrowth'
                elif self.method == "MLPRegressor": 
                    out_var = 'forestAge'
                
                #preds["{out_var}_pred".format(out_var = out_var)] = xr.DataArray([self.denorm_target(y_hat) if self.method=='MLPRegressor' else y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                #preds["{out_var}_obs".format(out_var = out_var)] = xr.DataArray([self.denorm_target(Y_cluster[mask_nan]) if self.method=='MLPRegressor' else Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                
                preds["{out_var}_pred".format(out_var = out_var)] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["{out_var}_obs".format(out_var = out_var)] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                save_cube.update_cube(preds.transpose('sample', 'cluster'), initialize=True)
   
    def denorm_target(self, 
                      x: np.array) -> np.array:
        """
        Returns de-normalized target. 
        
        The last dimension of `x` must match the length of `self.target_norm_stats`.
        
        Parameters:
            x: np.array
                The array to de-normalize.
        
        Returns:
            np.array: The de-normalized array.
        """
        
        return x * self.mldata.norm_stats[self.DataConfig['target'][0]]['std'] + self.mldata.norm_stats[self.DataConfig['target'][0]]['mean']
                

    