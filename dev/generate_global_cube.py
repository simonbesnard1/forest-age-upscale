#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:46:13 2022

@author: simon
"""
#%% Load modules
from ageUpscaling.global_cube.global_cube import GlobalCube

#%% Generate cube
cube_gen = GlobalCube('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/global_product',
                      '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/global_cube_gen/config_global_cube.yaml')
cube_gen.generate_cube()