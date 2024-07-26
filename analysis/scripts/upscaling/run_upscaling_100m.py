#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
"""
#%%Load modules
from ageUpscaling.upscaling.upscaling import UpscaleAge
from ageUpscaling.diagnostic.report import Report
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
print(f'Number of jobs requested is {SLURM_NTASKS}')

SLURM_ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

#%% Run upscaling
DataConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/upscaling/100m/data_config_xgboost.yaml"
UpscaleConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/upscaling/100m/config_upscaling.yaml"
upscale_init = UpscaleAge(DataConfig_path = DataConfig_path,
	                   upscaling_config_path= UpscaleConfig_path,
	                   algorithm = 'XGBoost',
	                   exp_name  = 'Age_upscale_100m',
	                   base_dir= '/home/besnard/projects/forest-age-upscale/output/upscaling/',
	                   study_dir= '/home/besnard/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.2',
	                   n_jobs = SLURM_NTASKS)
	                  
upscale_init.ForwardRun(task_id=SLURM_ARRAY_TASK_ID)

