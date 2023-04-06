#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.methods.extrapolation_index import ExtrapolationIndex
from ageUpscaling.diagnostic.report import Report

#%% Run upscaling
DataConfig_path= "/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/extrapolation_index/data_config.yaml"
CubeConfig_path= "/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/extrapolation_index/config_EI_cube.yaml"
EI_init = ExtrapolationIndex(DataConfig_path = DataConfig_path,
            	                   cube_config_path= CubeConfig_path,
                                   base_dir = '/home/simon/gfz_hpc/projects/forest-age-upscale/output/ExtrapolationIndex',
            	                   n_jobs = 5)
EI_init.calculate_global_index()

#%% Generate report
report_ = Report(study_dir= '/home/simon/gfz_hpc/projects/forest-age-upscale/output/ExtrapolationIndex/')
report_.generate_diagnostic(diagnostic_type =  {'extrapolation-index'})


