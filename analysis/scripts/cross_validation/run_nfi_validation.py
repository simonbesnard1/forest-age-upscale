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
                        
#%% Generate cross-validation report
report_ = Report(study_dir= '/home/besnard/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1',
		  nfi_data = '/home/besnard/projects/forest-age-upscale/data/training_data/nfi_valid_data.csv',
                 dist_cube = '/home/besnard/projects/forest-age-upscale/data/cubes/LandsatDisturbanceTime_100m')
report_.generate_diagnostic(diagnostic_type =  {'nfi-valid'})
