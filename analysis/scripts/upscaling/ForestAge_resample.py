#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.upscaling import UpscaleAge
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE"))
print(f'Number of jobs requested is {SLURM_NTASKS}')

#%% Run calculation
DataConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/upscaling/100m/data_config_xgboost.yaml"
UpscaleConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/upscaling/100m/config_upscaling.yaml"
upscale_init = UpscaleAge(DataConfig_path = DataConfig_path,
	                   upscaling_config_path= UpscaleConfig_path,
	                   algorithm = 'XGBoost',
	                   exp_name  = 'Age_upscale_100m',
	                   base_dir= '/home/besnard/projects/forest-age-upscale/output/upscaling/',
                           study_dir= '/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0',
	                   n_jobs = SLURM_NTASKS)

upscale_init.ParallelResampling(n_jobs=5)
