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
from ageUpscaling.utils.plotting import map_age_class

#%% Specify data directory
data_dir = '/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/'

#%% These should be replaced with your actual age data arrays/matrices for 2010 and 2020
out = []
out_group = []
for member_ in np.arange(20):
    age_class_2010 = xr.open_zarr(os.path.join(data_dir,'AgeClass_1deg')).sel(time = '2010-01-01', members=member_) 
    age_class_2020 = xr.open_zarr(os.path.join(data_dir,'AgeClass_1deg')).sel(time = '2020-01-01', members=member_)
    forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction
    
    # Earth's radius in kilometers
    EARTH_RADIUS = 6371.0
    
    # Calculate the width of each longitude slice in radians
    # Multiply by Earth's radius to get the width in kilometers
    delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    width_of_longitude = EARTH_RADIUS * delta_lon
    
    # Calculate the height of each latitude slice in radians
    # Multiply by Earth's radius to get the height in kilometers
    delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    height_of_latitude = EARTH_RADIUS * delta_lat
    
    # Now, calculate the area of each pixel in square kilometers
    # cos(latitude) factor accounts for the convergence of meridians at the poles
    # We need to convert latitude from degrees to radians first
    pixel_area = (width_of_longitude * height_of_latitude *
                  np.cos(np.deg2rad(age_class_2010.latitude))).broadcast_like(age_class_2010.isel(age_class=0))
    
    # Initialize a dictionary to hold the total area for each age class
    total_area_per_age_class_2010 = {}
    total_area_per_age_class_2020 = {}
    
    # Iterate over each age class, calculate the total area, and store it in the dictionary
    for age_class in age_class_2010.age_class.values:
        # Multiply the age fraction by the pixel area and sum over all pixels
        total_area_per_age_class_2010[age_class] = (age_class_2010.sel(age_class = age_class).forest_age * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
        total_area_per_age_class_2020[age_class] = (age_class_2020.sel(age_class = age_class).forest_age * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
        
    # Creating a DataFrame
    df = pd.DataFrame({
        'member': member_,
        'age_class': list(total_area_per_age_class_2010.keys()),
        'area_2010': list(total_area_per_age_class_2010.values()),
        'area_2020': list(total_area_per_age_class_2020.values()),
    })
    
    df['diff_area'] = df['area_2020'] - df['area_2010']
    df['rel_diff_area'] = np.round((df['diff_area'] / df['area_2010']) *100, 2)
    out.append(df.copy())
    
    df['age_class'] = df['age_class'].apply(map_age_class) 
    df_grouped = df.groupby('age_class').sum(numeric_only=True).reset_index()
    df_grouped['member'] = member_
    
    df_grouped['diff_area'] = df_grouped['area_2020'] - df_grouped['area_2010']
    df_grouped['rel_diff_area'] = np.round((df_grouped['diff_area'] / df_grouped['area_2010']) *100, 2)
    out_group.append(df_grouped)

#%% Compute statistics
out= pd.concat(out)
median_out = out.groupby("age_class").median(numeric_only=True)
q5_out = out.groupby("age_class").quantile(numeric_only=True, q=0.05)
q95_out = out.groupby("age_class").quantile(numeric_only=True, q=0.95)

out_group= pd.concat(out_group)
median_out = out_group.groupby("age_class").median(numeric_only=True)
q5_out = out_group.groupby("age_class").quantile(numeric_only=True, q=0.05)
q95_out = out_group.groupby("age_class").quantile(numeric_only=True, q=0.95)

