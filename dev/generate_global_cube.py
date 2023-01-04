#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:46:13 2022

@author: simon
"""
#%% Load modules
from ageUpscaling.global_cube.global_cube import GlobalCube
import time

if __name__ == '__main__':  

    start_time = time.time()

    cube_gen = GlobalCube('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/0d00083_annual/ESA_CCI_BIOMASS',
                          '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/global_cube_gen/config_global_cube.yaml')
    cube_gen.generate_cube()
    
    print("--- %s seconds ---" % (time.time() - start_time))
    

