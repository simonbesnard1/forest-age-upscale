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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor


def tune_(X_train, Y_train):
    
    param_grid = {'hidden_layer_sizes': [(50,50,50), (100,100,100), (200,200,200), 
                                         (50,50,50,50), (100,100,100,100), (200,200,200,200)],
                  'activation': ['relu'],
                  'solver': ['adam'],
                  'learning_rate': ['constant'],
                  'learning_rate_init': [0.001],
                  'power_t': [0.5],
                  'alpha': [0.0001],
                  'max_iter': [10000],
                  'early_stopping': [True],
                  'validation_fraction':[0.3],
                  'batch_size': [16,32,64],
                  'warm_start': [False]}
    model = MLPRegressor()  
    gridsearch = RandomizedSearchCV(estimator = model, param_distributions = param_grid, 
                                    n_iter = 10, verbose=0, random_state=42, n_jobs = 1)
    gridsearch.fit(X_train, Y_train)

    #Retrieve best model and best parameters
    best_model = gridsearch.best_estimator_
    
    return best_model

def predict_(model, X):    
    pred=model.predict(X)
    return(pred)
    
        
    
    
    