#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   MLPregression.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for training and optimizing a MLP regressor
"""

#%% Load lbrary
from sklearn.neural_network import MLPRegressor
from ageUpscaling.methods.ml_base import MLMethod

class MLPregression(MLMethod):
    def __init__(self, 
                hyper_params: dict = {},
                save_dir: str = './'):
        
        """MLPregression(tune_dir:str='/home/simon/')
        
        Method for for training and optimizing a MLP regressor.

        Parameters
        ----------
        tune_dir : str, default is "/home/simon/"
        
            string defining the directory for saving the model training results
        
        """
        
        super().__init__(
            save_dir=save_dir)
    
        self.model = MLPRegressor(
                                hidden_layer_sizes=(hyper_params['first_layer_neurons'], 
                                                    hyper_params['second_layer_neurons']),
                                learning_rate_init=hyper_params['learning_rate_init'],
                                activation=hyper_params['activation'],
                                batch_size=hyper_params['batch_size'],
                                random_state=1,
                                max_iter=100, 
                                early_stopping= True, 
                                validation_fraction = 0.3
                                )
        
            
        
    
    
    
