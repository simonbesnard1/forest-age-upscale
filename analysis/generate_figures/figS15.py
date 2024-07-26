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
from ageUpscaling.utils.plotting import violins

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

#%% Load age data
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction
average_age = xr.open_zarr(os.path.join(data_dir,'ForestAge_1deg')).forest_age.mean(dim = 'time').median(dim =  'members')
AgeDiff_1deg =  xr.open_zarr(os.path.join(data_dir,'AgeDiff_1deg')).median(dim = 'members')
stand_replaced_diff = AgeDiff_1deg.stand_replaced_diff
stand_replaced_diff = stand_replaced_diff.where(stand_replaced_diff < 0)

#%% Load management type
management_fraction = xr.open_zarr(os.path.join(data_dir,'ManagementTypeFrac_1deg')).where(forest_fraction >0)
agroforestry = management_fraction.agroforestry
intact_forests = management_fraction.intact_forests
naturally_regenerated = management_fraction.naturally_regenerated
oil_palm = management_fraction.oil_palm
plantation_forest = management_fraction.plantation_forest
planted_forest = management_fraction.planted_forest
logging_fraction = (management_fraction[["naturally_regenerated", 'plantation_forest']].to_array().sum(dim='variable')) / management_fraction.to_array().sum(dim='variable')

#%% Plot data
AgeBins = np.arange(0, 1.1, .1)
fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)
for j in range(len(AgeBins)-1):
    Agemask = (logging_fraction.values.reshape(-1) > AgeBins[j]) & (logging_fraction.values.reshape(-1) <= AgeBins[j+1])
    #Agemask = (age_windows_2010 > AgeBins[j]) & (age_windows_2010 <= AgeBins[j+1])            
    NEE_masked = np.abs(stand_replaced_diff.values.reshape(-1)[Agemask])
    #NEE_masked = NEE_windows_2010[Agemask]
    NEE_masked = NEE_masked[np.isfinite(NEE_masked)]
    IQ_mask = (NEE_masked < np.quantile(NEE_masked, 0.75)) & (NEE_masked > np.quantile(NEE_masked, 0.25))
    positive_values = NEE_masked[IQ_mask]
    
    # Calculate points for positive and negative values
    pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
    
    # Plot positive values in red
    ax[0].scatter(pointy_pos, pointx_pos, color='#8da0cb', alpha=0.2, marker='.')
    
    # Plot the mean as a large diamond
    ax[0].scatter(j, np.nanquantile(NEE_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
    
ax[0].set_xticks(np.arange(10))
ax[0].set_xticklabels(['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1'], rotation=0, size=14)
#ax1.set_xlabel('Age class', size=12)   
ax[0].set_xlabel('Logging fraction [adimensional]', size=14)
ax[0].set_ylabel('Age pre-stand-replacement [years]', size=14)

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].text(0.05, 1.1, 'a', transform=ax[0].transAxes,
            fontsize=16, fontweight='bold', va='top')
ax[0].tick_params(labelsize=12, rotation=90)

for j in range(len(AgeBins)-1):
    Agemask = (logging_fraction.values.reshape(-1) > AgeBins[j]) & (logging_fraction.values.reshape(-1) <= AgeBins[j+1])
    NEE_masked = np.abs(average_age.values.reshape(-1)[Agemask])
    NEE_masked = NEE_masked[np.isfinite(NEE_masked)]
    IQ_mask = (NEE_masked < np.quantile(NEE_masked, 0.75)) & (NEE_masked > np.quantile(NEE_masked, 0.25))
    positive_values = NEE_masked[IQ_mask]
    
    # Calculate points for positive and negative values
    pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
    
    # Plot positive values in red
    ax[1].scatter(pointy_pos, pointx_pos, color='#66c2a5', alpha=0.2, marker='.')
    
    # Plot the mean as a large diamond
    ax[1].scatter(j, np.nanquantile(NEE_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
    
ax[1].set_xticks(np.arange(10))
ax[1].set_xticklabels(['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1'], rotation=0, size=14)
#ax1.set_xlabel('Age class', size=12)   
ax[1].set_xlabel('Logging fraction [adimensional]', size=14)
ax[1].set_ylabel('Forest age [years]', size=14)

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].text(0.05, 1.1, 'b', transform=ax[1].transAxes,
            fontsize=16, fontweight='bold', va='top')
ax[1].tick_params(labelsize=12, rotation=90)
plt.savefig(os.path.join(plot_dir,'figS15.png'), dpi=300)



