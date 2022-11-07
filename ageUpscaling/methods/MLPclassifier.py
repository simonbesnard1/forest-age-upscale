#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:12:20 2020

@author: simon
"""
#%% Load lbrary
from sklearn.neural_network import MLPClassifier
import optuna
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class model_opti:
    def __init__(self, X_train, Y_train, iter_, tune_dir):
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.iter_ = iter_
        self.tune_dir = tune_dir
        
    def tune_(self):
                
        x_train, x_valid, y_train, y_valid = train_test_split(self.X_train, self.Y_train, random_state=1, train_size=0.3)

        if not os.path.exists(self.tune_dir + '/save_model/MLPClassifier/fold' + str(self.iter_)):
            os.makedirs(self.tune_dir + '/save_model/MLPClassifier/fold' + str(self.iter_))
        
        study = optuna.create_study(study_name = 'XvalAge_fold' + str(self.iter_), 
                                    storage='sqlite:///' + self.tune_dir + '/save_model/MLPClassifier/fold' + str(self.iter_) + '/hp_trial.db',
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                   reduction_factor=4, 
                                                                                   min_early_stopping_rate=0),
                                    direction='minimize')
        study.optimize(lambda trial: self._objective(trial, x_train, x_valid, y_train, y_valid, self.iter_, self.tune_dir), 
                       n_trials=300, n_jobs=4)
        
        with open(self.tune_dir + '/save_model/MLPClassifier/fold' + str(self.iter_) + '/model_trial' + "{}.pickle".format(study.best_trial.number), "rb") as fin:
            final_model = pickle.load(fin)
        
        
        return final_model
    
    def _objective(self, trial: optuna.Trial, x_train, x_valid, y_train, y_valid, iter_, tune_dir):
        
        params = {
            'learning_rate_init': trial.suggest_float('learning_rate_init ', 0.0001, 0.1, step=0.005),
            'first_layer_neurons': trial.suggest_int('first_layer_neurons', 10, 100, step=10),
            'second_layer_neurons': trial.suggest_int('second_layer_neurons', 10, 100, step=10),
            'activation': trial.suggest_categorical('activation', ['identity', 'tanh', 'relu']),
            'batch_size': trial.suggest_int('batch_size', 16, 64, step=16),
        }
    
        model = MLPClassifier(
            hidden_layer_sizes=(params['first_layer_neurons'], params['second_layer_neurons']),
            learning_rate_init=params['learning_rate_init'],
            activation=params['activation'],
            batch_size=params['batch_size'],
            random_state=1,
            max_iter=100, 
            early_stopping= True, 
            validation_fraction = 0.2
        )
    
        model.fit(x_train, y_train)
        
        with open(tune_dir + '/save_model/MLPClassifier/fold' + str(iter_) + '/model_trial' + "{}.pickle".format(trial.number), "wb") as fout:
            pickle.dump(model, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return log_loss(y_valid, model.predict(x_valid))
        

def predict_(model, X):    
    pred=model.predict(X)
    return(pred)
    
        
    
    
    
