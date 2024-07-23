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
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import os
from ageUpscaling.utils.plotting import calculate_pixel_area

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

#%% Load and process data
average_age = xr.open_zarr(os.path.join(data_dir,'ForestAge_1deg')).forest_age.mean(dim = 'time').median(dim =  'members')
age_difference = xr.open_zarr(os.path.join(data_dir,'AgeDiff_1deg')).age_difference.median(dim =  'members')
age_difference = age_difference -10
pixel_area = calculate_pixel_area(age_difference, 
                                  EARTH_RADIUS = 6371.0, 
                                  resolution=1)

#%% Calculate total area per age class for a given year
counts_2010 = []
counts_2020 = []

for member_ in np.arange(20):
    age_class_2010 = xr.open_zarr(os.path.join(data_dir,'AgeClass_1deg')).sel(time = '2010-01-01', members=member_) 
    age_class_2020 = xr.open_zarr(os.path.join(data_dir,'AgeClass_1deg')).sel(time = '2020-01-01', members=member_)
    forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction
    
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
    
#%% Plot data
age_classes = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-140',
               '141-160', '161-180', '181-200', '201-300', '>300']

fig = plt.figure(figsize=(14, 8), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)

# Plot 1: Average forest age
cbar_kwargs1 = dict(orientation='vertical', shrink=0.8, aspect=40, pad=0.04, spacing='proportional',
                    label='Forest age [years]', ticks=[300, 250, 200, 150, 100, 50])

ax_map1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
image1 = average_age.plot.imshow(ax=ax_map1, transform=ccrs.PlateCarree(), cmap='gist_earth_r', cbar_kwargs=cbar_kwargs1)
ax_map1.coastlines()
ax_map1.gridlines()
ax_map1.set_title('Average forest age 2010-2020', fontweight='bold')
ax_map1.text(0.05, 1.05, 'a', transform=ax_map1.transAxes, fontsize=16, fontweight='bold', va='top')

cbar1 = image1.colorbar
cbar1.set_ticks([300, 250, 200, 150, 100, 50])
cbar1.set_ticklabels([">300", "250", "200", "150", "100", "50"])

# Plot 2: Area distribution for 2010 and 2020
ax_scatter1 = fig.add_subplot(2, 2, 2)
width = 0.3
x = np.arange(len(age_classes))

median_counts_2010 = np.median(counts_2010, axis=0)
quantile_5_counts_2010 = np.percentile(counts_2010, 5, axis=0)
quantile_95_counts_2010 = np.percentile(counts_2010, 95, axis=0)

median_counts_2020 = np.median(counts_2020, axis=0)
quantile_5_counts_2020 = np.percentile(counts_2020, 5, axis=0)
quantile_95_counts_2020 = np.percentile(counts_2020, 95, axis=0)

error_bars_2010 = np.vstack((median_counts_2010 - quantile_5_counts_2010, quantile_95_counts_2010 - median_counts_2010))
error_bars_2020 = np.vstack((median_counts_2020 - quantile_5_counts_2020, quantile_95_counts_2020 - median_counts_2020))

ax_scatter1.bar(x - width/2, median_counts_2010, width, label='2010', color='blue', yerr=error_bars_2010, capsize=2)
ax_scatter1.bar(x + width/2, median_counts_2020, width, label='2020', color='green', yerr=error_bars_2020, capsize=2)

ax_scatter1.set_xticks(x)
ax_scatter1.set_xticklabels(age_classes, rotation=90)
ax_scatter1.spines['top'].set_visible(False)
ax_scatter1.spines['right'].set_visible(False)
ax_scatter1.set_ylabel('Area [billion hectares]', size=14)
ax_scatter1.legend(frameon=False, fontsize=14)
ax_scatter1.text(0.05, 1.05, 'b', transform=ax_scatter1.transAxes, fontsize=16, fontweight='bold', va='top')

# Plot 3: Age difference map
ax_map2 = fig.add_subplot(2, 2, 3, projection=ccrs.Robinson())
cbar_kwargs2 = {'shrink': 0.8, 'aspect': 40, 'pad': 0.04, 'ticks': [-30, -20, -10, 0], 'label': 'Age difference [years]'}

image2 = age_difference.plot.imshow(ax=ax_map2, transform=ccrs.PlateCarree(), cmap='afmhot', vmin=-30, vmax=0, cbar_kwargs=cbar_kwargs2)

cbar2 = image2.colorbar
cbar2.set_ticks([-30, -20, -10, 0])
cbar2.set_ticklabels(['>30', '-20', '-10', '0'])

ax_map2.coastlines()
ax_map2.gridlines()
ax_map2.set_title('Anomaly difference map \n relative to the expected 10-year aging', fontweight='bold')
ax_map2.text(0.05, 1.05, 'c', transform=ax_map2.transAxes, fontsize=16, fontweight='bold', va='top')
plt.savefig(os.path.join(plot_dir, 'fig1.png'), dpi=300)
