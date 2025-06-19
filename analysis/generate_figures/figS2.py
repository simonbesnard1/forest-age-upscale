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
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
from ageUpscaling.utils.plotting import load_and_plot_satellite_data, plot_forest_age_diff

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
data_dir = '/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-2.0/'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

# Load forest age dataset
ds = xr.open_zarr(os.path.join(data_dir,'AgeUpscale_100m')).forest_age.isel(members=10)
ds.attrs['long_name'] = 'Forest age change [years]'

# Set up plot
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), constrained_layout=True)

# Define regions of interest
regions = {
    "Cost Range, USA - Forest harvest and regrowth": ([-123.4, 43.7, -123.3, 43.8], [43.8, 43.7], [-123.4, -123.3], "2020-01-01", "2010-01-01"),
    #"Southern Rookies, USA - Wildfire": ([-108.3, 37.3, -108.2, 37.4], [37.4, 37.3], [-108.3, -108.2], "2020-01-01", "2010-01-01"),
    #"Les Landes, FR - Wildfire": ([0.1, 43.9, 0.2, 44], [0.1, 0.2], [43.9, 44], "2020-01-01", "2010-01-01"),
    "Scandinavian Peninsula - Plantation": ([14.8, 60.2, 14.9, 60.3], [60.3, 60.2], [14.8, 14.9], "2020-01-01", "2010-01-01"),
    "Amazon, BR - Secondary forest regrowth": ([-52.3, -3, -52.2, -2.9], [-2.9, -3], [-52.3, -52.2], "2020-01-01", "2010-01-01"),
    #"Central African Forests, Congo Basin, BR - Selective logging": ([24.3, 0.18, 24.7, 0.21], [0.18, 0.21], [24.3, 24.7], "2020-01-01", "2010-01-01"),
                                        
}

# Iterate over regions and plot data
fig = plt.figure(figsize=(12, 6 * len(regions)),constrained_layout= True)

# Create a GridSpec for the entire figure
outer_grid = gridspec.GridSpec(len(regions), 1, figure=fig)

# Iterate over regions
for i, (region, (bbox, lat_range, lon_range, time1, time2)) in enumerate(regions.items()):
    # Create a nested 1x2 grid for each region within the outer grid
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i])

    # Create subplots
    ax1 = fig.add_subplot(inner_grid[0])
    ax2 = fig.add_subplot(inner_grid[1])

    # Plot satellite data in the first subplot
    load_and_plot_satellite_data(bbox, ax1)
    
    # Plot forest age difference in the second subplot
    plot_forest_age_diff(ds, lat_range, lon_range, time1, time2, ax2)

    # Set titles and aspect ratio
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax1.set_title('', fontweight='bold')
    
    # Set x-axis and y-axis label
    ax1.set_xlabel('longitude [degrees east]')
    ax2.set_xlabel('longitude [degrees east]')
    ax1.set_ylabel('latitude [degrees east]')
    ax2.set_ylabel('latitude [degrees east]')
    
    ax_middle = fig.add_subplot(outer_grid[i], frame_on=False)
    ax_middle.set_xticks([])
    ax_middle.set_yticks([])
    ax_middle.set_title(f'{region}', fontweight='bold', fontsize=18)
    
plt.savefig(os.path.join(plot_dir,'figS2.png'), dpi=300)
