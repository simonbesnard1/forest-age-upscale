#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:11:52 2023

@author: simon
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# These should be replaced with your actual age data arrays/matrices for 2010 and 2020
age_2010 = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/ForestAge_fraction_1deg').sel(time = '2010-01-01') 
age_2020 = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/ForestAge_fraction_1deg').sel(time = '2020-01-01')
forest_fraction = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/ForestFraction_1deg').forest_fraction

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
              np.cos(np.deg2rad(age_2010.latitude))).broadcast_like(age_2010.isel(age_class=0))

# Initialize a dictionary to hold the total area for each age class
total_area_per_age_class_2010 = {}
total_area_per_age_class_2020 = {}

# Iterate over each age class, calculate the total area, and store it in the dictionary
for age_class in age_2010.age_class.values:
    # Multiply the age fraction by the pixel area and sum over all pixels
    total_area_per_age_class_2010[age_class] = (age_2010.sel(age_class = age_class).forest_age * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
    total_area_per_age_class_2020[age_class] = (age_2020.sel(age_class = age_class).forest_age * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
    

# Example age classes and counts for 2010 and 2020
age_classes = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140',
               '140-160', '160-180', '180-200', '>200']

# These should be your actual counts of forest areas for each age class in 2010 and 2020.
counts_2010 = np.stack(total_area_per_age_class_2010.values())
counts_2020 = np.stack(total_area_per_age_class_2020.values())

#%% Plot data
width = 0.35  # the width of the bars
fig, ax = plt.subplots(1,1, figsize=(5,3),  gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)

# Plotting both years' data
x = np.arange(len(age_classes))  # the label locations
rects1 = ax.bar(x - width/2, counts_2010, width, label='2010', color='blue')
rects2 = ax.bar(x + width/2, counts_2020, width, label='2020', color='green')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xticks(x)
ax.set_xticklabels(age_classes, rotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Forest age classes [years]')
ax.set_ylabel('Area [billion hectares]', size=12)
ax.legend(frameon=False, fontsize=12, loc="best")
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS2.png', dpi=300)

