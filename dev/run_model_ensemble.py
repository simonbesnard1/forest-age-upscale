#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.transformers.model_ensemble import ModelEnsemble

#%% Run upscaling
CubeConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/upscaling/1km/config_ensemble_cube.yaml"
model_ensemble_init = ModelEnsemble(cube_config_path= CubeConfig_path,
                                 n_jobs = 50,
                                 study_dir= '/home/besnard/projects/forest-age-upscale/output/upscaling/Age_upscale_1km/XGBoost/version-1.1')
model_ensemble_init.calculate_global_index()
