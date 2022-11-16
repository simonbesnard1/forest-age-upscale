#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:10:01 2022

@author: simon
"""

#%% Load library
from ageUpscaling.methods.MLP import MLPmethod

#%% Load config files
cube_path = "/home/simon/Documents/science/GFZ/projects/forest_age_upscale/data/training_data/training_data_ageMap_OG300.nc"
train_subset= [1, 2, 3, 4]
valid_subset= [5, 6, 7, 8]
data_config_path= "/home/simon/Documents/science/GFZ/projects/forest_age_upscale/experiments/data_config.yaml"

#%% Run training
mlp_method = MLPmethod(save_dir="/home/simon/Documents/science/GFZ/projects/forest_age_upscale/output/test_train/", data_config_path= data_config_path)
mlp_method.train(cube_path=cube_path, train_subset=train_subset,valid_subset=valid_subset)
