#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:17:20 2023

@author: simon
"""

import xarray as xr
import pandas as pd
import glob

#%% Load files
worlClim_ = glob.glob('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/0d0083_static/WorlClim/*.nc')
training_data = pd.read_csv('/home/simon/gfz_hpc/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_v2.csv') 

for file_ in worlClim_:
    ds_ = xr.open_dataset(file_)
    clim_extract = []
    for index, row in training_data.iterrows():
        clim_extract.append(ds_.sel(latitude = row['latitude'], longitude = row['longitude'], method = 'nearest').to_array().values)
    
    extracted_df = pd.DataFrame(clim_extract, columns=[list(ds_.keys())[0]])

    training_data = pd.concat([training_data, extracted_df], axis=1)
training_data.to_csv('/home/simon/gfz_hpc/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_v2_withworlClim.csv')


    
    