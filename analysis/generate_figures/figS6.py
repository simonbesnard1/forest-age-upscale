#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
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
                        
#%% Generate cross-validation report
report_ = Report(study_dir= study_.study_dir)
report_.generate_diagnostic(diagnostic_type =  {'cross-validation'})