#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:57 2022

@author: simon
"""
from ageUpscaling.core.study import Study
from ageUpscaling.diagnostic.report import Report
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE"))

#%% Initiate experiment
DataConfig_path= "/home//besnard/projects/forest-age-upscale/config_files/cross_validation/data_config_xgboost.yaml"
CubeConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/cross_validation/config_prediction_cube.yaml"
study_ = Study(DataConfig_path = DataConfig_path,
               cube_config_path= CubeConfig_path,
               exp_name = 'subsetFIA_genetic',
               algorithm  = 'XGBoost',
               base_dir= '/home/besnard/projects/forest-age-upscale/output/cross_validation',
               n_jobs = SLURM_NTASKS)
study_.cross_validation(n_folds=30,
			 valid_fraction=0.3, 
                        feature_selection=True, 
                        feature_selection_method= 'genetic')
                        
#%% Generate cross-validation report
report_ = Report(study_dir= study_.study_dir)
report_.generate_diagnostic(diagnostic_type =  {'cross-validation'})
