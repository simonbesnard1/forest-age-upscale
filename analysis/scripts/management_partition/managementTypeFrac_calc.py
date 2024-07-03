#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.diagnostic.management_type import ManagementType
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE"))
SLURM_ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

#%% Run calculation
DataConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/management_partition/config_management_type.yaml"
calc_management = ManagementType(Config_path = DataConfig_path,
                                 study_dir= '/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0',
                                 n_jobs = SLURM_NTASKS)
calc_management.ManagementResample()

