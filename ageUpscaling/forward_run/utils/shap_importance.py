#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:00:00 2020

@author: simon
"""

#%% Load library
import shap
import xgboost as xgb

def shap_importance(model, X):
    """
    Compute permutation feature importances for scikit-learn models using
    a validation set.
    
    """
    #%% Compute model performance baseline model
    mybooster = model.get_booster()
    model_bytearray = mybooster.save_raw()[4:]
    def myfun(self=None):
        return model_bytearray
    mybooster.save_raw = myfun
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    #shap_interaction_values = explainer.shap_interaction_values(X)
    return shap_values#, shap_interaction_values

# def shap_importance(model, X,Y, n_features):
#     """
#     Compute permutation feature importances for scikit-learn models using
#     a validation set.
    
#     """
#     #%% Compute model performance baseline model
#     xg_importance = xgb.DMatrix(X.reshape(-1, n_features), label =  Y.reshape(-1))
#     shap_values = model.predict(xg_importance, pred_contribs= True)
#     return shap_values
