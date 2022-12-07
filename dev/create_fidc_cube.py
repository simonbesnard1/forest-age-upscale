#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:56:09 2022

@author: simon
"""
from ageUpscaling.fidc_cube.csv_to_fidcCube import ImportAndSave

#%% Run data cubing
fidc_prov = ImportAndSave(input_csv= '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_fullFIA.csv' ,
                          out_file='/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_fullFIA.nc')
data_ = fidc_prov.compute_cube(variables= 'default')
