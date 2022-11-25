#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:57 2022

@author: simon
"""
from ageUpscaling.core.study import Study

#%% Initiate experiment
DataConfig_path= "/home/simon/Documents/science/GFZ/projects/forest_age_upscale/experiments/data_config.yaml"
study_ = Study(DataConfig_path = DataConfig_path,
               study_name  = 'upscaling_MLPregressor',
               out_dir= '/home/simon/Documents/science/GFZ/projects/forest_age_upscale/output/',
               n_jobs = 10)
study_.model_training(n_model=10,
                       valid_fraction=0.5, 
                       feature_selection=True, 
                       feature_selection_method= 'recursive')