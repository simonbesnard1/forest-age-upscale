#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.upscaling.age_class_fraction import AgeFraction
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE"))
print(f'Number of jobs requested is {SLURM_NTASKS}')

#%% Run calculation
DataConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/age_class_fraction/config_age_fraction.yaml"
calc_age_fraction = AgeFraction(Config_path = DataConfig_path,
                                study_dir= '/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.2/',
                                n_jobs = SLURM_NTASKS)
calc_age_fraction.AgeClassCubeInit()
