#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:15:15 2022

@author: simon
"""
#%%Load modules
from ageUpscaling.upscaling.age_class_fraction import AgeFraction

#%% Run upscaling
DataConfig_path= "/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/age_class_fraction/config_age_fraction.yaml"
calc_age_fraction = AgeFraction(Config_path = DataConfig_path,
                                study_dir= '/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_1km/XGBoost/version-1.0/AgeUpscale_1km/')
calc_age_fraction.AgeFractionCalc()
