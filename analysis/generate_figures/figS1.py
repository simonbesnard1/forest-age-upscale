#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
"""
#%% load library
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import os
import matplotlib as mpl

# Set matplotlib parameters for consistent styling
params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
    'font.size': 16,
    # axes
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 14,
    # tex
    'text.usetex': True,
}

mpl.rcParams.update(params)

#%% Specify data and plot directories
data_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/data/Age_upscale_100m'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

# Load the data
mixed_effect_management = xr.open_dataset(os.path.join(data_dir,'mixed_effect_management.nc')).p_value_interaction

# Categorize significant and non-significant values
significant_ = mixed_effect_management < 0.01
non_significant_ = mixed_effect_management >= 0.01

# Create a combined DataArray for plotting
combined = xr.where(significant_, 1, xr.where(non_significant_, 0, np.nan))

# %%Plot data
fig, ax = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(11, 4), 
                       constrained_layout=True, gridspec_kw={'wspace': 0, 'hspace': 0})
image = mixed_effect_management.plot.imshow(ax=ax[0], transform=ccrs.PlateCarree(), vmax =0.01,
                                            cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                                                               label ='p-values'))
ax[0].coastlines()
ax[0].gridlines()
ax[0].text(0.05, 1.05, '(a)', transform=ax[0].transAxes,
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
ax[1].text(0.05, 1.05, '(b)', transform=ax[1].transAxes,
            fontsize=16, fontweight='bold', va='top')

# Save the figure
plt.savefig(os.path.join(plot_dir,'figS1.png'), dpi=300)


