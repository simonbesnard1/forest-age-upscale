#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
from ageUpscaling.utils.plotting import calculate_pixel_area

#%% Specify data and plot directories
data_dir = '/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/'

#%% Load forest fraction
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction
pixel_area = calculate_pixel_area(forest_fraction, 
                                  EARTH_RADIUS = 6378.160, 
                                  resolution=1)

#%% These should be replaced with your actual age data arrays/matrices for 2010 and 2020
out = []
for member_ in np.arange(20):
    stand_replaced_class_partition = xr.open_zarr(os.path.join(data_dir,'AgeDiffPartition_1deg')).sel(members=member_).stand_replaced_class_partition 
    aging_forest_class_partition = xr.open_zarr(os.path.join(data_dir,'AgeDiffPartition_1deg')).sel(members=member_).aging_forest_class_partition 
    
    
    # Initialize a dictionary to hold the total area for each age class
    total_area_stand_replaced = {}
    total_area_aging_forest = {}
    
    # Iterate over each age class, calculate the total area, and store it in the dictionary
    for age_class in stand_replaced_class_partition.age_class.values:
        # Multiply the age fraction by the pixel area and sum over all pixels
        total_area_stand_replaced[age_class] = (stand_replaced_class_partition.sel(age_class = age_class) * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
        total_area_aging_forest[age_class] = (aging_forest_class_partition.sel(age_class = age_class) * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
        
    # Creating a DataFrame
    df = pd.DataFrame({
        'member': member_,
        'age_class': list(total_area_stand_replaced.keys()),
        'area_stand_replaced': list(total_area_stand_replaced.values()),
        'area_aging_forest': list(total_area_aging_forest.values()),
    })
    
    # Calculate the total area for each column
    total_area_stand_replaced = df['area_stand_replaced'].sum()
    total_area_aging_forest = df['area_aging_forest'].sum()
    
    df['percent_area_stand_replaced'] = (df['area_stand_replaced'] / total_area_stand_replaced) * 100
    df['percent_area_aging_forest'] = (df['area_aging_forest'] / total_area_aging_forest) * 100
    
    out.append(df.copy())
    
#%% Compute statistics
out= pd.concat(out)
median_out = out.groupby("age_class").median(numeric_only=True)
q5_out = out.groupby("age_class").quantile(numeric_only=True, q=0.05)
q95_out = out.groupby("age_class").quantile(numeric_only=True, q=0.95)

