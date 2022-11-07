#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:32:44 2020

@author: simon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:12:20 2020

@author: simon
"""
#%% Load lbrary
import xgboost as xgb
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#import numpy as np
#from sklearn.utils import compute_sample_weight

         
def tune_(X_train, Y_train, X_valid, Y_valid):
    
    #age_weight = compute_sample_weight('balanced', np.round(Y_train, -1))
    
    # Build input matrices for XGBoost    
    params = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7, 0.9],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'n_estimators' : [300, 500],
        'gamma':[0,0.5,1],
        'objective': ['reg:squarederror'],
        'tree_method': ["hist"]
    }
           
    model = xgb.XGBRegressor(nthread = 1)        
    gridsearch = RandomizedSearchCV(model, params, return_train_score=False, n_iter=200)
    fit_params={'early_stopping_rounds':20,\
                'eval_metric': 'rmse',
                'eval_set':[(X_valid, Y_valid)]}
    gridsearch.fit(X_train, Y_train, verbose=0,
                   **fit_params)

    #Retrieve best model and best parameters
    #best_params = gridsearch.best_params_
    best_model_reg = gridsearch.best_estimator_
    
    return best_model_reg

def predict_(model_reg, X):    
    pred=model_reg.predict(X)
    return(pred)
    
        
    
    
    