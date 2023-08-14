#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:17:20 2023

@author: simon
"""

import rioxarray as rio 
import pandas as pd
import glob
import os
import numpy as np

#%% Load files
soilgrids_ = glob.glob('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/0d002083_static/soilgrids/v2_0_0/org_data/*/*.tif')
training_data = pd.read_csv('/home/simon/gfz_hpc/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_v2.csv') 

for file_ in soilgrids_:
    ds_ = rio.open_rasterio(file_).sel(band=1)
    soilgrids_extract = []
    for index, row in training_data.iterrows():
        extract_ = ds_.sel(y = row['latitude'], x = row['longitude'], method = 'nearest').values
        soilgrids_extract.append(ds_.sel(y = row['latitude'], x = row['longitude'], method = 'nearest').values)
    
    extracted_df = pd.DataFrame(soilgrids_extract, columns=[os.path.basename(file_).split('.tif')[0]])
    extracted_df[extracted_df<0] = np.nan

    training_data = pd.concat([training_data, extracted_df], axis=1)
training_data.to_csv('/home/simon/gfz_hpc/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_v2_withsoilgrids.csv')


    
    