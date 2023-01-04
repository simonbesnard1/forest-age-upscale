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

class FeatureSelection(ABC):
    """A class for selecting the most important features in a dataset.
    
    Parameters:
        method: str, optional (default='regression')
            The type of model to use for training, either 'regression' or 'classification'.
        feature_selection_method: str, optional (default='boruta')
            The method to use for feature selection, either 'boruta' or 'recursive'.
        features: dict, optional
            A dictionary of features to use in the model.
    """
    
    def __init__(self, 
                 method:str='regression',
                 feature_selection_method:str = "boruta",
                 features:dict = {}):
    
        self.method = method
        self.feature_selection_method = feature_selection_method
        self.features = features
        
    def get_features(self,
                     data:dict= {},
                     max_features:int=10,
                     max_depth:int=5,
                     n_jobs:int=1)-> np.array:
        """Selects the most important features from the input data using the specified feature selection method.
        
        Parameters
        ----------
        data : dict, default is {}
            Dictionary containing the 'features' and 'target' arrays for feature selection.
        
        max_features : int, default is 10        
            Maximum number of features to select.
        
        max_depth : int, default is 5
            Maximum depth of the model.
        
        n_jobs : int, default is -1
            Number of jobs to use for feature selection.
        
        Returns
        -------
        var_selected : np.array
            Array containing the list of selected features.
        """
        
        if "Regressor" in self.method:
            rf = RandomForestRegressor(n_jobs=n_jobs)
        elif "Classifier" in self.method:
            rf = RandomForestClassifier(n_jobs=n_jobs, class_weight='balanced', max_depth=max_depth)
         
        if self.feature_selection_method == "boruta":
            feat_selector = BorutaPy(rf, n_estimators='auto')
            feat_selector.fit(data['features'], data['target'])
        elif self.feature_selection_method == "recursive":
            feat_selector = RFE(rf, n_features_to_select=max_features, step=1)
            feat_selector.fit(data['features'], data['target'])
            
        var_selected = np.array(self.features)[feat_selector.support_]
            
        return var_selected
        





