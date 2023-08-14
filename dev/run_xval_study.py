#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:57 2022

@author: simon
"""
from ageUpscaling.core.study import Study
from ageUpscaling.diagnostic.report import Report

#%% Initiate experiment
DataConfig_path= "/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/cross_validation/data_config_xgboost.yaml"
CubeConfig_path= "/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/cross_validation/config_prediction_cube.yaml"
study_ = Study(DataConfig_path = DataConfig_path,
               cube_config_path= CubeConfig_path,
               exp_name = 'subsetFIA',
               algorithm  = 'XGBoost',
               base_dir= '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/output/',
               n_jobs = 15)
study_.cross_validation(n_folds=20, 
                        valid_fraction=0.3, 
                        feature_selection=True, 
                        feature_selection_method= 'recursive')
    
#%% Generate report
report_ = Report(study_dir= '/home/simon/gfz_hpc/projects/forest-age-upscale/output/cross_validation/subsetFIA_recursive/XGBoost/version-1.4')
report_.generate_diagnostic(diagnostic_type =  {'cross-validation'})
