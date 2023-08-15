#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:05:51 2023

@author: simon
"""

from ageUpscaling.cubegen.cube.soilgrids import SoilGrids

cube_test = SoilGrids(base_file= '/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/0d002083_static/soilgrids/v2_0_0/org_data/bdod/bdod_0-5cm_mean.tif', 
                     cube_config_path= '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/global_cube_gen/config_soilgrids_cube.yaml')
cube_test.fill_cube(var_name = 'bdod_0_5cm_mean', 
                    chunk_data=True,
                    n_workers =10)