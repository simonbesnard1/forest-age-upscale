#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:57 2022

@author: simon
"""
from ageUpscaling.core.experiment import Experiment

#%% Initiate experiment
DataConfig_path= "/home/simon/Documents/science/GFZ/projects/forest_age_upscale/experiments/data_config.yaml"
exp_ = Experiment(DataConfig_path = DataConfig_path,
                  exp_name  = 'MLPregressor',
                  base_dir= '/home/simon/Documents/science/GFZ/projects/forest_age_upscale/output/')
exp_.xval(n_folds=10, valid_fraction=0.3, feature_selection=False, prediction=True)
