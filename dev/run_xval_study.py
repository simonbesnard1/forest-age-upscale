#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:57 2022

@author: simon
"""
from ageUpscaling.core.study import Study

#%% Initiate experiment
DataConfig_path= "/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/cross_validation/data_config.yaml"
CubeConfig_path= "/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/cross_validation/config_prediction_cube.yaml"
study_ = Study(DataConfig_path = DataConfig_path,
               cube_config_path= CubeConfig_path,
               study_name  = 'Xval_ForestAge',
               base_dir= '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/output/',
               n_jobs = 10)
study_.cross_validation(n_folds=10, 
                        valid_fraction=0.5, 
                        feature_selection=True, 
                        feature_selection_method= 'recursive')
