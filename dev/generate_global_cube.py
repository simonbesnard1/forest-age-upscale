#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:46:13 2022

@author: simon
"""
#%% Load modules
from ageUpscaling.global_cube.global_cube import GlobalCube
import time
from dask.distributed import Client


if __name__ == '__main__':
    
    client = Client()
    

    start_time = time.time()
    
    #%% Generate cube
    cube_gen = GlobalCube('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/global_product',
                          '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/global_cube_gen/config_global_cube.yaml')
    cube_gen.generate_cube()
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    client.close()

