#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:53:15 2023

@author: simon
"""
#%% load library
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

# Load the data
mixed_effect_management = xr.open_dataset('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/mixed_effect_management.nc').p_value_interaction

# Categorize significant and non-significant values
significant_ = mixed_effect_management < 0.01
non_significant_ = mixed_effect_management >= 0.01

# Create a combined DataArray for plotting
combined = xr.where(significant_, 1, xr.where(non_significant_, 0, np.nan))

# Plot data
fig, ax = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(11, 4), 
                       constrained_layout=True, gridspec_kw={'wspace': 0, 'hspace': 0})
image = mixed_effect_management.plot.imshow(ax=ax[0], transform=ccrs.PlateCarree(), vmax =0.01,
                                            cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                                                               label ='p-values'))
ax[0].coastlines()
ax[0].gridlines()
ax[0].text(0.05, 1.05, 'a', transform=ax[0].transAxes,
            fontsize=16, fontweight='bold', va='top')


# Define colormap and normalization
cmap = plt.get_cmap('coolwarm', 2)  # Using a colormap with two discrete colors
norm = plt.Normalize(0, 1)

# Plot the combined data
image = combined.plot.imshow(ax=ax[1], transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)
cbar = plt.colorbar(image, ax=ax[1], orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional')
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(['Non-significant', 'Significant'])

ax[1].coastlines()
ax[1].gridlines()
ax[1].text(0.05, 1.05, 'b', transform=ax[1].transAxes,
            fontsize=16, fontweight='bold', va='top')

# Save the figure
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS16.png', dpi=300)


