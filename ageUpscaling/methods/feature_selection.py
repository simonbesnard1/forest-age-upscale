#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   feature_selection.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for feature selection
"""

#%%Load library
import xarray as xr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
import numpy as np
from sklearn.feature_selection import RFE

class FeatureSelection(object):
    
    def __init__(self, 
                 model:str='regression',
                 selection_method:str = "boruta"):
        """FeatureSelction(model:str='regression', selection_method:str = "boruta")
        
        Method for selecting most important features.

        Parameters
        ----------
        model : str, default is "regression"
        
            string defining the model type for training - "regression" or "classification"
            
        selection_method : str, default is "boruta"
        
            string defining the method used for features selection - "boruta" or "recursive"
            
        Return
        ------
        an array (np.array): an array containing the list of selected features
        
        """
        
        self.model = model
        self.selection_method = selection_method
        
    def get_features(self,
                     data:xr.Dataset= [],
                     features:dict= [],
                     target:dict= [],
                     max_age:int=300,
                     n_features:int=10,
                     max_depth:int=5,
                     n_jobs:int=10):
        
        """Parameters
        ----------
        n_features : int, default is "10        
            integer defining the maximum number of features to select
        
        
        n_features : int, default is "10        
            integer defining the maximum number of features to select
            
        max_depth : int, default is 5
            integer defining the maximum depth of the model
        
        n_jobs : int, default is 10
            integer defining the number of jobs
            
        """
        X = data[features].to_array().transpose('cluster', 'sample', 'variable').values
        Y = data[target].values
        
        if self.model == "regression":
            rf = RandomForestRegressor(n_jobs=n_jobs)
        elif self.model == "classification":
            rf = RandomForestClassifier(n_jobs=n_jobs, class_weight='balanced', max_depth=max_depth)
            Y[Y==max_age] = 1
            Y[Y<max_age] = 0
            
        n_features = X.shape[2]
        X, Y = X.reshape(-1, n_features), Y.reshape(-1)
        mask_train = (np.all(np.isfinite(X), axis=1)) & (np.isfinite(Y))
        X, Y = X[mask_train, :], Y[mask_train]
        
        if self.selection_method == "boruta":
            feat_selector = BorutaPy(rf, n_estimators='auto')
            feat_selector.fit(X, Y)
        elif self.selection_method == "recursive":
            feat_selector = RFE(rf, n_features_to_select=10, step=1)
            feat_selector.fit(X, Y)
            
        var_selected = np.array(features)[feat_selector.support_]
        X_filtered = feat_selector.transform(X)
            
        return {"var_selected" : var_selected, "X_filtered": X_filtered}
        





