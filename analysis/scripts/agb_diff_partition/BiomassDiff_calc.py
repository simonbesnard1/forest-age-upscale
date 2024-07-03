#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.diagnostic.biomassDiff_partition import BiomassDiffPartition
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE"))
SLURM_ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

#%% Run calculation
DataConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/agb_diff_partition/config_agbDiff_partition.yaml"
calc_agb_diff = BiomassDiffPartition(Config_path = DataConfig_path,
                             		 study_dir= '/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0',
                              	n_jobs = SLURM_NTASKS)
calc_agb_diff.BiomassDiffPartitionCalc(task_id=SLURM_ARRAY_TASK_ID)

