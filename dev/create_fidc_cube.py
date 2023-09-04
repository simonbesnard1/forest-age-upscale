#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:56:09 2022

@author: simon
"""
from ageUpscaling.fidc_cube.csv_to_fidcCube import ImportAndSave
import numpy as np

#%% Run data cubing
fidc_prov = ImportAndSave(input_csv= '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_subsetFIA_v5.csv',
                          out_file='/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_subsetFIA_v5.nc')
data_ = fidc_prov.compute_cube(variables= 'default')
cluster_ = data_.cluster.values
np.save('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/xval_index_subsetFIA_v3.npy', cluster_)

