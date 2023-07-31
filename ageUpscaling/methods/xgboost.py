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
from typing import Any, Tuple

import xarray as xr

import xgboost as xgb
from sklearn.metrics import mean_squared_error, roc_auc_score

import optuna

from ageUpscaling.dataloaders.ml_dataloader import MLDataModule
#from ageUpscaling.utils.metrics import mef_gufunc
from ageUpscaling.methods.feature_selection import FeatureSelection

def quantile_loss(y_true, y_pred, quantiles):
    losses = []
    for quantile in quantiles:
        residual = y_true - y_pred
        loss_q = np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)
        losses.append(loss_q)
    return np.sum(losses)

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))

def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))


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
                 tune_dir: str=None,
                 DataConfig:dict=None,
                 method:str = 'XGBoostRegressor') -> None:

        self.tune_dir = tune_dir
        
        if not os.path.exists(tune_dir):
            os.makedirs(tune_dir)
            
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
              task_:str= '',
              feature_selection:bool= True,
              feature_selection_method:str='',
              biais_correction:bool= True,
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
        if feature_selection:
            self.DataConfig['features_selected'] = FeatureSelection(method=task_, 
                                                                    feature_selection_method = feature_selection_method, 
                                                                    features = self.DataConfig['features'],
                                                                    data = xr.open_dataset(self.DataConfig['training_dataset'])).get_features(n_jobs = n_jobs)
        else:
            self.DataConfig['features_selected'] = self.DataConfig['features'].copy()
        

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
                                    #storage='sqlite:///' + self.tune_dir + '/trial_model/hp_trial.db',
                                    # pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                    #                                                reduction_factor=4, 
                                    #                                                min_early_stopping_rate=8),
                                    direction=['minimize' if self.method == 'XGBoostRegressor' else 'maximize'][0])
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
                   retrain_with_valid:bool= True) -> float:
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
                        }
        
        training_params = {'num_boost_round': trial.suggest_int('num_boost_round', DataConfig['hyper_params']['num_boost_round']['min'], DataConfig['hyper_params']['num_boost_round']['max'], step=DataConfig['hyper_params']['num_boost_round']['step'])}
        training_params['early_stopping_rounds'] = 1000
        
        if self.method == "XGBoostRegressor":
            #hyper_params['objective'] = "reg:squarederror"
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-rmse")

            
        elif self.method == "XGBoostClassifier":
            hyper_params['objective'] = "binary:logistic"
            hyper_params['eval_metric'] = "auc"      
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "eval-auc")

        dtrain = xgb.DMatrix(train_data['features'], label=train_data['target'])
        deval = xgb.DMatrix(val_data['features'], label = val_data['target'])
        vallist = [(dtrain, 'train'), (deval, 'eval')]
        
        if self.method == "XGBoostRegressor":
        
            model_ = xgb.train(hyper_params, dtrain, evals=vallist, callbacks = [pruning_callback],
                                custom_metric = rmsle, obj= squared_log, verbose_eval=False, **training_params)
        elif self.method == "XGBoostClassifier":
            
            model_ = xgb.train(hyper_params, dtrain, evals=vallist, callbacks = [pruning_callback],
                               verbose_eval=False, **training_params)

            
        self.best_ntree = model_.best_ntree_limit
            
        if retrain_with_valid:
            training_params['num_boost_round'] = self.best_ntree
            training_params['early_stopping_rounds'] = None
            if self.method == "XGBoostRegressor":
                
                model_ = xgb.train(hyper_params, dtrain, custom_metric = rmsle, obj= squared_log, **training_params)

            elif self.method == "XGBoostClassifier":
                   
                model_ = xgb.train(hyper_params, dtrain, **training_params)

        with open(tune_dir + "/trial_model/model_trial_{id_}.pickle".format(id_ = trial.number), "wb") as fout:
            pickle.dump(model_, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if self.method == "XGBoostRegressor":
            #loss_ =   mean_squared_error(val_data['target'], model_.predict(xgb.DMatrix(val_data['features'])), squared=False) #/ (np.max(val_data['target']) - np.min(val_data['target']))
            #loss_ = quantile_loss(val_data['target'], model_.predict(xgb.DMatrix(val_data['features'])), [0.05, 0.25, 0.5, 0.75, 0.95])
            loss_ =   mean_squared_error(val_data['target'], model_.predict(xgb.DMatrix(val_data['features'])), squared=False) #/ (np.max(val_data['target']) - np.min(val_data['target']))
            
            #loss_ += 1 - mef_gufunc(val_data['target'], model_.predict(xgb.DMatrix(val_data['features'])))
        elif self.method == "XGBoostClassifier":
            loss_ =  roc_auc_score(val_data['target'], np.rint(model_.predict(xgb.DMatrix(val_data['features']))))
        
        return loss_
    
    def predict_clusters(self, 
                        save_cube:str) -> None:
        """Predict the target variables for the given data and save the results in a prediction zarr cube.
    
        Parameters:
            save_cube: str
                Path to the output netCDF file where the predictions will be saved.
            biais_correction : bool
                Whether to apply a biais correction or not.
                Default is True.
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
                    y_hat =  np.rint(self.best_model.predict(dpred))
                
                preds = xr.Dataset()
                
                if self.method == "XGBoostClassifier": 
                    out_var = 'oldGrowth'
                elif self.method == "XGBoostRegressor": 
                    out_var = 'forestAge'
                
                preds["{out_var}_pred".format(out_var = out_var)] = xr.DataArray([y_hat], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                preds["{out_var}_obs".format(out_var = out_var)] = xr.DataArray([Y_cluster[mask_nan]], coords = {'cluster': [self.mldata.test_subset[cluster_]], 'sample': np.arange(len(y_hat))})
                
                save_cube.update_cube(preds.transpose('sample', 'cluster'), initialize=True)
   
    
    