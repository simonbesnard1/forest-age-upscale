#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:00:00 2020

@author: simon
"""

#%% Load library
import shap
import xgboost as xgb

def shap_importance(model, X_train, X_test):
    """
    Compute permutation feature importances for scikit-learn models using
    a validation set.
    
    """
    #%% Compute model performance baseline model
    explainer = shap.KernelExplainer(model.predict,shap.sample(X_train, 1000))
    shap_values = explainer.shap_values(X_test)
    return shap_values#, shap_interaction_values
