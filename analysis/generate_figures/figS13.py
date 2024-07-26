#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
"""
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
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

#%% These should be replaced with your actual age data arrays/matrices for 2010 and 2020
age_2020 = xr.open_zarr(os.path.join(data_dir,'ForestAge_fraction_1deg')).sel(time = '2020-01-01')
age_2060_scenario1 = xr.zeros_like(age_2020)

#%% Define a mapping from the 2020 age classes to the 2060 age classes
age_class_mapping = {
    '0-20': '20-40',
    '20-40': '40-60',
    '40-60': '60-80',
    '60-80': '80-100',
    '80-100': '100-120',
    '100-120': '120-140',
    '120-140': '140-160',
    '140-160': '160-180',
    '160-180': '180-200',  # Assuming >180 becomes >200
    '180-200': '>200',  # Assuming >180 becomes >200
    '>200': '>200'      # Assuming >200 remains >200
}

for old_age_class, new_age_class in age_class_mapping.items():
    age_2060_scenario1.forest_age.loc[:,:,new_age_class] += age_2020.forest_age.loc[:,:,old_age_class]
        
# Define the midpoint for each age class
age_midpoints = {
    '0-20': 10,
    '20-40': 30,
    '40-60': 50,
    '60-80':70,
    '80-100':90,
    '100-120':110,
    '120-140':130,
    '140-160':150,
    '160-180':170,
    '180-200':190,
    '>200':200
}

# Initialize an empty DataArray for the weighted sum
weighted_mean_age_2020 = xr.zeros_like(age_2020.isel(age_class=0))
weighted_mean_age_2060_scenario1 = xr.zeros_like(age_2060_scenario1.isel(age_class=0))

# Iterate over the age classes and calculate the weighted sum
for age_class, midpoint in age_midpoints.items():
    weighted_mean_age_2020 += age_2020.sel(age_class = age_class) * midpoint
    weighted_mean_age_2060_scenario1 += age_2060_scenario1.sel(age_class = age_class) * midpoint
weighted_mean_age_2020 = weighted_mean_age_2020.where(weighted_mean_age_2020>0)
weighted_mean_age_2060_scenario1 = weighted_mean_age_2060_scenario1.where(weighted_mean_age_2060_scenario1>0)

#%% Setup the figure and axes for a 2x2 grid
fig, ax = plt.subplots(2,2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(9, 7), constrained_layout=True)
cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                   label ='Forest age [years]')

image = weighted_mean_age_2020.forest_age.plot.imshow(ax=ax[0,0], transform=ccrs.PlateCarree(), cmap='gist_earth_r',
                           cbar_kwargs=cbar_kwargs)
ax[0,0].coastlines()
ax[0,0].gridlines()
ax[0,0].set_title('Forest age in 2050 \n under BAU scenario')
ax[0,0].text(0.05, 1.05, 'a', transform=ax[0,0].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = weighted_mean_age_2060_scenario1.forest_age.plot.imshow(ax=ax[0,1], transform=ccrs.PlateCarree(), cmap='gist_earth_r',
                                           cbar_kwargs=cbar_kwargs)
ax[0,1].coastlines()
ax[0,1].gridlines()
ax[0,1].set_title('Forest age in 2060 \n under conservation scenario')
ax[0,1].text(0.05, 1.05, 'b', transform=ax[0,1].transAxes,
            fontsize=16, fontweight='bold', va='top')

diff_2020_scenario1 = weighted_mean_age_2060_scenario1 - weighted_mean_age_2020
image = diff_2020_scenario1.forest_age.plot.imshow(ax=ax[1,1], transform=ccrs.PlateCarree(), cmap='Blues',
                           cbar_kwargs=cbar_kwargs)
ax[1,1].coastlines()
ax[1,1].gridlines()
ax[1,1].set_title('Conservation - BAU')
ax[1,1].text(0.05, 1.05, 'c', transform=ax[1,1].transAxes,
            fontsize=16, fontweight='bold', va='top')
fig.delaxes(ax[1,0])
plt.savefig(os.path.join(plot_dir,'figS13.png'), dpi=300)
