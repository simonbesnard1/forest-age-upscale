#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:12:20 2020

@author: simon
"""
#%% Load lbrary

import xarray as xr
import numpy as np
import sys
sys.path.append('/Net/Groups/BGI/work_2/FIDC_age_upscale/code/Xval_model/MLP')
from utils import MLPclassifier, MLPclassifier_optuna
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold
        
#%% Load forest age data        
print('Loading data')
age_data = xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/training_data/training_data_ageMap_OG300_new.nc")

#%% Create feature and target arrays
feature_ = np.concatenate(pd.read_csv('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/feature_selection/variable_selected_borutaClass.csv').values)
#age_data['agb_exp'] = np.exp(age_data['agb'])
#X = age_data[feature_].to_array().transpose('cluster', 'sample', 'variable').values
#Y = age_data['age'].values
#Y_class = Y.copy()
#Y_class[Y==300] = 1
#Y_class[Y<300] = 0
#lats = np.repeat(age_data.latitude.values, Y_class.shape[1]).reshape(-1, Y_class.shape[1])
#lons = np.repeat(age_data.longitude.values, Y_class.shape[1]).reshape(-1, Y_class.shape[1])
                            
#%% Split data for CV and remove missing values
print('Splitting data and remove missing values')
cv_pred= []  
cv_obs= []  
shap_values = []
shap_feature = []
n_features = len(feature_)
lats_ = []
lons_ = []
cv = np.unique(age_data.cluster.values.reshape(-1))
cv = cv[np.isfinite(cv)]
cv = np.unique(age_data.cluster.values.reshape(-1))
cv = cv[np.isfinite(cv)]
np.random.shuffle(cv)
kf = KFold(n_splits=10)
iter_ = 1
for train_index, test_index in kf.split(cv):
    print(f'fold {iter_}')
    #X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    #lats_test, lons_test = lats[test_index], lons[test_index]
    #lats_train, lons_train = lats[train_index], lons[train_index]    
    #X_train, Y_train, X_test, Y_test = X_train.reshape(-1, n_features), Y_train.reshape(-1), X_test.reshape(-1, n_features), Y_test.reshape(-1)
    test_set = age_data.where(np.isin(age_data.cluster, cv[test_index]))
    train_set = age_data.where(np.isin(age_data.cluster, cv[train_index]))
    X_train = train_set[feature_].to_array().transpose('plot', 'sample', 'variable').values
    Y_train = train_set['age'].values
    X_test = test_set[feature_].to_array().transpose('plot', 'sample', 'variable').values
    Y_test = test_set['age'].values
    X_train, Y_train, X_test, Y_test = X_train.reshape(-1, n_features), Y_train.reshape(-1), X_test.reshape(-1, n_features), Y_test.reshape(-1)    
    mask_train = (np.all(np.isfinite(X_train), axis=1)) & (np.isfinite(Y_train))
    X_train, Y_train = X_train[mask_train, :], Y_train[mask_train]
    mask_test = (np.all(np.isfinite(X_test), axis=1)) & (np.isfinite(Y_test))
    X_test, Y_test = X_test[mask_test, :], Y_test[mask_test]    
    lats_test, lons_test = test_set['latitude_origin'].values.reshape(-1)[mask_test], test_set['longitude_origin'].values.reshape(-1)[mask_test]
    
    #%% Train random forest
    print('Model training and inference')
    if Y_test.shape[0]>0:
        print('scaling data')   
        scaler = StandardScaler()
        scaler.fit(X_train)
        StandardScaler(copy=True, with_mean=True, with_std=True)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        print('Model training and inference')
        Y_class_train = Y_train.copy()
        Y_class_train[Y_train==300] = 1
        Y_class_train[Y_train<300] = 0
        Y_class_test = Y_test.copy()
        Y_class_test[Y_test==300] = 1
        Y_class_test[Y_test<300] = 0        
        best_model = MLPclassifier_optuna.model_opti(X_train, Y_class_train, iter_, '/Net/Groups/BGI/work_2/FIDC_age_upscale/output/Xval_results/MLP/optuna').tune_()
        #best_model = MLPregression.tune_(X_train, Y_train)     
        #y_hat = MLPregression.predict_(best_model, X_test)
        #best_model_class = MLPclassifier.tune_(X_train, Y_class_train)
        y_hat = MLPclassifier_optuna.predict_(best_model, X_test)
        cv_pred.append(y_hat)
        cv_obs.append(Y_class_test)    
        lats_.append(lats_test)
        lons_.append(lons_test)
        iter_ += 1    
            
#%% Export prediction
print('Exporting cross-validation results')
cv_obs = np.concatenate(cv_obs)
cv_pred = np.concatenate(cv_pred)
lats_ = np.concatenate(lats_)
lons_ = np.concatenate(lons_)
cv_output = xr.Dataset({'obs': xr.DataArray(
                            data   = cv_obs,   
                            dims   = ['sample'],
                            coords = {'sample': np.arange(cv_obs.shape[0])}),
                        'pred': xr.DataArray(
                            data   = cv_pred,   
                            dims   = ['sample'],
                            coords = {'sample': np.arange(cv_obs.shape[0])}),
                        'lat': xr.DataArray(
                            data   = lats_,   
                            dims   = ['sample'],
                            coords = {'sample': np.arange(cv_obs.shape[0])}),
                        'lon': xr.DataArray(
                            data   = lons_,   
                            dims   = ['sample'],
                            coords = {'sample': np.arange(cv_obs.shape[0])})})       
cv_output.to_netcdf('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/Xval_results/MLP/Xval_output_MLPclass_OG300.nc', mode='w')    
