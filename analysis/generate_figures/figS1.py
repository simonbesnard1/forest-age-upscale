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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os

# Set matplotlib parameters for consistent styling
params = {
    # font
    'font.family': 'serif',
    'font.size': 14,
    # axes
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 12
}

mpl.rcParams.update(params)

#%% Specify data and plot directories
data_dir = '/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

#%% Load forest fraction
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction

#%% Load transcom regions
GFED_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/GFED_regions/GFED_regions_360_180_v1.nc').basis_regions
GFED_regions = GFED_regions.where((GFED_regions == 9) | (GFED_regions == 8))
GFED_regions = GFED_regions.where((GFED_regions ==9) | (np.isnan(GFED_regions)), 5)
GFED_regions = GFED_regions.where((GFED_regions ==5) | (np.isnan(GFED_regions)), 6)
GFED_regions = GFED_regions.rename({'lat' : 'latitude', 'lon' : 'longitude'})
transcom_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/transcom_regions/transcom_regions_360_180.nc').transcom_regions
transcom_regions = transcom_regions.reindex(latitude=transcom_regions.latitude[::-1])
transcom_regions = transcom_regions.where(transcom_regions<=11)
transcom_regions = transcom_regions.where((transcom_regions<5) | (transcom_regions>6) )
transcom_regions = transcom_regions.where(np.isfinite(transcom_regions), GFED_regions)
transcom_regions['latitude'] = forest_fraction['latitude']
transcom_regions['longitude'] = forest_fraction['longitude']

transcom_mask ={"class_7":{"eco_class" : 7, "name": "Eurasia Boreal"},                
                "class_1":{"eco_class":  1, "name": "NA Boreal"},
                "class_8":{"eco_class" : 8, "name": "Eurasia Temperate"},
                "class_11":{"eco_class" : 11, "name": "Europe"},                
                "class_2":{"eco_class" : 2, "name": "NA Temperate"},
                "class_4":{"eco_class" : 4, "name": "SA Temperate"},
                "class_3":{"eco_class" : 3, "name": "SA Tropical"},
                "class_9":{"eco_class" : 9, "name": "Tropical Asia"},
                "class_5":{"eco_class" : 5, "name": "Northern Africa"},
                "class_6":{"eco_class" : 6, "name": "Southern Africa"},
                "class_10":{"eco_class" : 10, "name": "Australia"}}

#%% These should be replaced with your actual age data arrays/matrices for 2010 and 2020
area_2010_region = {}
area_2020_region = {}

for region_ in list(transcom_mask.keys()):
    
    class_values = transcom_mask[region_]['eco_class']
    class_name = transcom_mask[region_]['name']
    
    counts_2010 = []
    counts_2020 = []
    
    for member_ in np.arange(20):
        age_class_2010 = xr.open_zarr(os.path.join(data_dir,'AgeClass_1deg')).sel(time = '2010-01-01', members=member_) 
        age_class_2010 = age_class_2010.where(transcom_regions==class_values)
        age_class_2020 = xr.open_zarr(os.path.join(data_dir,'AgeClass_1deg')).sel(time = '2020-01-01', members=member_)
        age_class_2020 = age_class_2020.where(transcom_regions==class_values)
        
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
            
        # These should be your actual counts of forest areas for each age class in 2010 and 2020.
        counts_2010.append(list(total_area_per_age_class_2010.values()))
        counts_2020.append(list(total_area_per_age_class_2020.values()))
    area_2010_region[class_name] = counts_2010
    area_2020_region[class_name] = counts_2020
    
    
#%% Setup the figure and axes for a 2x2 grid
age_classes = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-140',
               '141-160', '161-180', '181-200', '201-300', '>300']

fig, axes = plt.subplots(4, 3, figsize=(16, 16), gridspec_kw={'wspace': 0, 'hspace': 0.1}, constrained_layout=True)
axes = axes.flatten()

for i, region_ in enumerate(transcom_mask.keys()):
    class_name = transcom_mask[region_]['name']

    ax = axes[i]
    
    width = 0.3  # the width of the bars
    
    median_counts_2010 = np.median(area_2010_region[class_name], axis=0)
    quantile_5_counts_2010 = np.percentile(area_2010_region[class_name], 5, axis=0)
    quantile_95_counts_2010 = np.percentile(area_2010_region[class_name], 95, axis=0)

    median_counts_2020 = np.median(area_2020_region[class_name], axis=0)
    quantile_5_counts_2020 = np.percentile(area_2020_region[class_name], 5, axis=0)
    quantile_95_counts_2020 = np.percentile(area_2020_region[class_name], 95, axis=0)

    # Calculate the error bars
    error_bars_2010 = np.vstack((median_counts_2010 - quantile_5_counts_2010,
                            quantile_95_counts_2010 - median_counts_2010))
    error_bars_2020 = np.vstack((median_counts_2020 - quantile_5_counts_2020,
                            quantile_95_counts_2020 - median_counts_2020))

    x = np.arange(len(age_classes))  # the label locations
    
    rects1 = ax.bar(x - width/2, median_counts_2010, width, label='2010', 
                             color='blue', yerr=error_bars_2010, capsize=4)
    rects2 = ax.bar(x + width/2, median_counts_2020, width, label='2020', 
                             color='green', yerr=error_bars_2020, capsize=4)
    
    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(age_classes, rotation=90)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Area [billion hectares]', size=14)
    ax.legend(frameon=False, fontsize=14)
    ax.set_title(class_name, fontweight='bold')
    ax.text(0.05, 1.05, chr(97 + i), transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
fig.delaxes(axes[-1])
plt.savefig(os.path.join(plot_dir,'figS1.png'), dpi=300)
