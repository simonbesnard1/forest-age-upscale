#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:22:16 2023

@author: simon
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
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
    'legend.fontsize': 12,
    # tex
    #'text.usetex': True,
    # layout
    #'constrained_layout': True
}

mpl.rcParams.update(params)


#%% Temporary age calc
average_age = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestAge_1deg').forest_age.mean(dim = 'time').median(dim =  'members')

# Calculate the difference
age_difference = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiff_1deg').age_difference.median(dim =  'members')
age_difference = age_difference -10
#age_difference = age_difference.where(np.isfinite(weighted_mean_age_2010))
#age_difference_relative = (age_difference / weighted_mean_age_2010.values) *100

#%% These should be replaced with your actual age data arrays/matrices for 2010 and 2020
counts_2010 = []
counts_2020 = []

for member_ in np.arange(20):
    age_class_2010 = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeClass_1deg').sel(time = '2010-01-01', members=member_) 
    age_class_2020 = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeClass_1deg').sel(time = '2020-01-01', members=member_)
    forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction
    
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
    
#%% Setup the figure and axes for a 2x2 grid
# Example age classes and counts for 2010 and 2020
age_classes = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-140',
               '141-160', '161-180', '181-200', '201-300', '>300']

fig = plt.figure(figsize=(14,8), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)

cbar_kwargs = dict(orientation='vertical', shrink=0.8, aspect=40, pad=0.04, spacing='proportional',
                   label ='Forest age [years]', ticks= [300, 250, 200, 150, 100, 50])

ax_map1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
image = average_age.plot.imshow(ax=ax_map1, transform=ccrs.PlateCarree(), cmap='gist_earth_r', cbar_kwargs=cbar_kwargs)
ax_map1.coastlines()
ax_map1.gridlines()
ax_map1.set_title('Average forest age 2010-2020', fontweight='bold')
ax_map1.text(0.05, 1.05, 'a', transform=ax_map1.transAxes,
            fontsize=16, fontweight='bold', va='top')

cbar = image.colorbar
cbar.set_ticks([300, 250, 200, 150, 100, 50])  # Define your tick positions
cbar.set_ticklabels([">300", "250", "200", "150", "100", "50"])  # Set custom labels


ax_scatter1 = fig.add_subplot(2, 2, 2)

#x = np.arange(len(list(class_ranges.keys())))  # the label locations
width = 0.3  # the width of the bars

x = np.arange(len(age_classes))  # the label locations

# Calculate the median, 5th percentile, and 95th percentile
median_counts_2010 = np.median(counts_2010, axis=0)
quantile_5_counts_2010 = np.percentile(counts_2010, 5, axis=0)
quantile_95_counts_2010 = np.percentile(counts_2010, 95, axis=0)

median_counts_2020 = np.median(counts_2020, axis=0)
quantile_5_counts_2020 = np.percentile(counts_2020, 5, axis=0)
quantile_95_counts_2020 = np.percentile(counts_2020, 95, axis=0)

# Calculate the error bars
error_bars_2010 = np.vstack((median_counts_2010 - quantile_5_counts_2010,
                        quantile_95_counts_2010 - median_counts_2010))
error_bars_2020 = np.vstack((median_counts_2020 - quantile_5_counts_2020,
                        quantile_95_counts_2020 - median_counts_2020))

rects1 = ax_scatter1.bar(x - width/2, median_counts_2010, width, label='2010', 
                         color='blue', yerr=error_bars_2010, capsize=2)
rects2 = ax_scatter1.bar(x + width/2, median_counts_2020, width, label='2020', 
                         color='green', yerr=error_bars_2020, capsize=2)

# Add some text for labels, title, and custom x-axis tick labels
ax_scatter1.set_xticks(x)
ax_scatter1.set_xticklabels(age_classes, rotation=90)

#bars1 = ax_scatter1.bar(x - width/2, counts_2010.values(), width, label='2010', color='blue')
#bars2 = ax_scatter1.bar(x + width/2, summarized_areas_2020.values(), width, label='2020', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax_scatter1.spines['top'].set_visible(False)
ax_scatter1.spines['right'].set_visible(False)
ax_scatter1.set_ylabel('Area [billion hectares]', size=14)
#ax_scatter1.set_xticks(x)
#ax_scatter1.set_xticklabels(list(class_ranges.keys()), size=12)
ax_scatter1.legend(frameon=False, fontsize=14)
ax_scatter1.text(0.05, 1.05, 'b', transform=ax_scatter1.transAxes,
            fontsize=16, fontweight='bold', va='top')

ax_map2 = fig.add_subplot(2, 2, 3, projection=ccrs.Robinson())

# Color bar customization
cbar_kwargs = {
    'shrink': 0.8,
    'aspect': 40,
    'pad': 0.04,
    'ticks': [-30, -20, -10, 0],  # Define ticks according to your data range
    'label': 'Age difference [years]'
}

# Plotting
image = age_difference.plot.imshow(ax=ax_map2, transform=ccrs.PlateCarree(), cmap='afmhot', vmin=-30, vmax=0,
                            cbar_kwargs=cbar_kwargs)

# Custom color bar labels, including ">30"
cbar = image.colorbar
cbar.set_ticks([-30, -20, -10, 0])  # Define your tick positions
cbar.set_ticklabels(['>30', '-20', '-10', '0'])  # Set custom labels

# Additional map features
ax_map2.coastlines()
ax_map2.gridlines()
ax_map2.set_title('Anomaly difference map \n relative to the expected 10-year aging', fontweight='bold')
ax_map2.text(0.05, 1.05, 'c', transform=ax_map2.transAxes,
            fontsize=16, fontweight='bold', va='top')


# image = age_difference.plot.imshow(ax=ax_map2, transform=ccrs.PlateCarree(), cmap='afmhot', vmin=-30, vmax=0,
#                             cbar_kwargs=cbar_kwargs)
# ax_map2.coastlines()
# ax_map2.gridlines()
# ax_map2.set_title('Anomaly difference map \n relative to the expected 10-year aging', fontweight='bold')
# ax_map2.text(0.05, 1.05, 'c', transform=ax_map2.transAxes,
#             fontsize=16, fontweight='bold', va='top')
# ax_map3 = fig.add_subplot(2, 2, 4, projection=ccrs.Robinson())

# cbar_kwargs['label'] = 'Forest age [%]'
# image = age_difference_relative.plot.imshow(ax=ax_map3, transform=ccrs.PlateCarree(), cmap='seismic_r', vmin=-35, vmax=35,
#                            cbar_kwargs=cbar_kwargs)
# ax_map3.coastlines()
# ax_map3.gridlines()
# ax_map3.set_title('Relative difference map [2020 - 2010]', fontweight='bold')
# ax_map3.text(0.05, 1.05, 'd', transform=ax_map3.transAxes,
#             fontsize=16, fontweight='bold', va='top')
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/fig1.png', dpi=300)
