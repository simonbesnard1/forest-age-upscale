#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:46:13 2022

@author: simon
"""
#%% Load modules
from ageUpscaling.global_cube.global_cube import GlobalCube
import time

start_time = time.perf_counter()

#%% Generate cube
cube_gen = GlobalCube('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/global_product',
                      '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/global_cube_gen/config_global_cube.yaml')
cube_gen.generate_cube()

end_time = time.perf_counter()

processing_time = end_time - start_time

print(f"Time taken to process script: {processing_time:.2f} seconds")
