#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:57 2022

@author: simon
"""
from ageUpscaling.core.study import Study
from ageUpscaling.diagnostic.report import Report
                        
#%% Generate cross-validation report
report_ = Report(study_dir= '/home/besnard/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1',
		  nfi_data = '/home/besnard/projects/forest-age-upscale/data/training_data/nfi_valid_data.csv',
                 dist_cube = '/home/besnard/projects/forest-age-upscale/data/cubes/LandsatDisturbanceTime_100m')
report_.generate_diagnostic(diagnostic_type =  {'nfi-valid'})
