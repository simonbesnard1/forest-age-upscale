#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:46:13 2022

@author: simon
"""
from ageUpscaling.global_cube.global_cube import GlobalCube

#%% Generate cube
cube_gen = GlobalCube(base_file_path = '/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/0d0083_static/WorlClim/',
                      cube_config_path= '/home/simon/gfz_hpc/projects/forest-age-upscale/config_files/worlclim_cube_gen/config_global_cube.yaml')
cube_gen.generate_cube()	
    

    

