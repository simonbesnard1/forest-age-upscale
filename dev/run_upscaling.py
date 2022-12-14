#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.upscaling.upscaling import UpscaleAge

#%% Run upscaling
DataConfig_path= "/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/forward_run/data_config.yaml"
CubeConfig_path= "/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/forward_run/config_prediction_cube.yaml"
upscale_init = UpscaleAge(DataConfig_path = DataConfig_path,
                           cube_config_path= CubeConfig_path,
                           study_name  = 'test_upscale',
                           base_dir= '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/output/',
                           n_jobs = 10)
upscale_init.ForwardRun()
