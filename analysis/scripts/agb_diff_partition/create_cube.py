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
from ageUpscaling.diagnostic.biomassDiff_partition import BiomassDiffPartition
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE"))
print(f'Number of jobs requested is {SLURM_NTASKS}')

#%% Run calculation
DataConfig_path= "/home/besnard/projects/forest-age-upscale/config_files/agb_diff_partition/config_agbDiff_partition.yaml"
calc_agb_diff = BiomassDiffPartition(Config_path = DataConfig_path,
                             		 study_dir= '/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.2',
                              	n_jobs = SLURM_NTASKS)
calc_agb_diff.BiomassDiffPartitionCubeInit()
