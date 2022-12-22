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

#%%Load library
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
import numpy as np
from sklearn.feature_selection import RFE

class FeatureSelection(object):
    
    def __init__(self, 
                 method:str='regression',
                 feature_selection_method:str = "boruta",
                 features:dict = {}):
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
        
        self.method = method
        self.feature_selection_method = feature_selection_method
        self.features = features
        
    def get_features(self,
                     data:dict= {},
                     max_features:int=10,
                     max_depth:int=5,
                     n_jobs:int=-1):
        
        """Parameters
        ----------
        n_features : int, default is "10        
            integer defining the maximum number of features to select
        
        
        n_features : int, default is "10        
            integer defining the maximum number of features to select
            
        max_depth : int, default is 5
            integer defining the maximum depth of the model
        
        n_jobs : int, default is -1
            integer defining the number of jobs
            
        """
        
        if self.method == "MLPRegressor":
            rf = RandomForestRegressor(n_jobs=n_jobs)
        elif self.method == "MLPClassifier":
            rf = RandomForestClassifier(n_jobs=n_jobs, class_weight='balanced', max_depth=max_depth)
         
        if self.feature_selection_method == "boruta":
            feat_selector = BorutaPy(rf, n_estimators='auto')
            feat_selector.fit(data['features'], data['target'])
        elif self.feature_selection_method == "recursive":
            feat_selector = RFE(rf, n_features_to_select=max_features, step=1)
            feat_selector.fit(data['features'], data['target'])
            
        var_selected = np.array(self.features)[feat_selector.support_]
            
        return var_selected
        





