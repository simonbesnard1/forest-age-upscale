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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import numpy as np

def tune_(X_train, Y_train):
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
    model = RandomForestRegressor()  
    gridsearch = RandomizedSearchCV(estimator = model, param_distributions = random_grid, 
                                    n_iter = 50, verbose=0, random_state=42, n_jobs = 1)
    gridsearch.fit(X_train, Y_train)

    #Retrieve best model and best parameters
    best_model = gridsearch.best_estimator_

    #n_features   = X_train.shape[1]
    #max_features = n_features//3
    #RandomForestRegressor_kwargs={'n_estimators':300, 'oob_score':False,
    #                            'verbose':0, 'warm_start':False,
    #                            'n_jobs':1,'max_features':max_features}
    #best_model=RandomForestRegressor(**RandomForestRegressor_kwargs)
    #best_model.fit(X_train,Y_train)
    
    return best_model

def predict_(model, X):    
    pred=model.predict(X)
    return(pred)
    
        
    
    
    