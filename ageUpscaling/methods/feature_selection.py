#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   feature_selection.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for feature selection
"""

import numpy as np

from abc import ABC

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE
from boruta import BorutaPy

import xarray as xr

class FeatureSelection(ABC):
    """A class for selecting the most important features in a dataset.
    
    Parameters:
        method: str, optional (default='regression')
            The type of model to use for training, either 'regression' or 'classification'.
        feature_selection_method: str, optional (default='boruta')
            The method to use for feature selection, either 'boruta' or 'recursive'.
        features: dict, optional
            A dictionary of features to use in the model.
        data : xr.Dataset, default is None
            xr.Dataset containing the training dataset
    """
    
    def __init__(self, 
                 method:str='regression',
                 feature_selection_method:str = "boruta",
                 features:dict = {},
                 data:xr.Dataset = None):
    
        self.method = method
        self.feature_selection_method = feature_selection_method
        self.features = features
        self.data = data
        
    def get_features(self,
                     max_forest_age:int= 300,
                     max_features:int=10,
                     max_depth:int=5,
                     n_jobs:int=1)-> np.array:
        """Selects the most important features from the input data using the specified feature selection method.
        
        Parameters
        ----------        
        max_features : int, default is 10        
            Maximum number of features to select.
        
        max_depth : int, default is 5
            Maximum depth of the model.
        
        n_jobs : int, default is 1
            Number of jobs to use for feature selection.
        
        Returns
        -------
        var_selected : np.array
            Array containing the list of selected features.
        """
        X = self.data[self.features].to_array().transpose('cluster','sample', 'variable').values.reshape(-1, len(self.features))
        Y = self.data[['age']]
        
        if 'Classifier' in self.method :
            Y = Y.to_array().values.reshape(-1)
            mask_old = Y==max_forest_age
            mask_young = Y<max_forest_age
            Y[mask_old] = 1
            Y[mask_young] = 0    
             
        else :
            Y = Y.where(Y<max_forest_age).to_array().values.reshape(-1)
            Y[Y<1] = 1 ## set min age to 1
            
        mask_nan = (np.all(np.isfinite(X), axis=1)) & (np.isfinite(Y))
        
        X, Y = X[mask_nan, :], Y[mask_nan]   
        
        if "Regressor" in self.method:
            rf = RandomForestRegressor(n_jobs=n_jobs)
        elif "Classifier" in self.method:
            rf = RandomForestClassifier(n_jobs=n_jobs, class_weight='balanced', max_depth=max_depth)
        
        if self.feature_selection_method == "boruta":
            feat_selector = BorutaPy(rf, n_estimators='auto', perc=80)

        elif self.feature_selection_method == "recursive":
            feat_selector = RFE(rf, n_features_to_select=max_features, step=1)
        
        feat_selector.fit(X, Y)
            
        var_selected = list(np.array(self.features)[feat_selector.support_])
            
        return var_selected
        





