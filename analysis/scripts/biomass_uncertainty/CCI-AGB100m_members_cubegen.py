#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
"""
from ageUpscaling.upscaling.biomass_uncertainty import BiomassUncertainty
import glob
import os

# Retrieve the value of the environment variable
SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
print(f'Number of jobs requested is {SLURM_NTASKS}')
SLURM_ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

biomass_uncertainty = BiomassUncertainty(Config_path= '/home/besnard/projects/forest-age-upscale/config_files/ESABiomass_cube_gen/config_members_cube.yaml',
					  study_dir= '/home/besnard/projects/forest-age-upscale/data/cubes/')
biomass_uncertainty.BiomassUncertaintyCalc(task_id=SLURM_ARRAY_TASK_ID)
