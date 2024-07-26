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

#%% Load library
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

#%% Load data
BiomassDiffPartition_1deg =  xr.open_zarr(os.path.join(data_dir,'BiomassPartition_1deg'))
Young_stand_aging_class =  (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '0-20') * 0.47 ) 
Intermediate_stand_aging_class = (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '20-80') * 0.47 ) 
Mature_stand_aging_class =   (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '80-200') * 0.47)
OG_stand_aging_class =   (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '>200')* 0.47)

#%% Plot data
fig, ax = plt.subplots(2,2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(10, 8), constrained_layout=True, gridspec_kw={'wspace': 0, 'hspace': 0})
cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                   label ='Aboveground carbon [MgC ha$^{-1}$]')

image = Young_stand_aging_class.plot.imshow(ax=ax[0,0], transform=ccrs.PlateCarree(), cmap='YlGnBu', vmin=0, vmax=100,
                           cbar_kwargs=cbar_kwargs)
ax[0,0].coastlines()
ax[0,0].gridlines()
ax[0,0].set_title('Gradually ageing young forests \n (1-20 years)')
ax[0,0].text(0.05, 1.05, 'a', transform=ax[0,0].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = Intermediate_stand_aging_class.plot.imshow(ax=ax[0,1], transform=ccrs.PlateCarree(), cmap='YlGnBu', vmin=0, vmax=100,
                                           cbar_kwargs=cbar_kwargs)
ax[0,1].coastlines()
ax[0,1].gridlines()
ax[0,1].set_title('Gradually ageing maturing forests \n (21-80 years)')
ax[0,1].text(0.05, 1.05, 'b', transform=ax[0,1].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = Mature_stand_aging_class.plot.imshow(ax=ax[1,0], transform=ccrs.PlateCarree(), cmap='YlGnBu', vmin=0, vmax=100,
                           cbar_kwargs=cbar_kwargs)
ax[1,0].coastlines()
ax[1,0].gridlines()
ax[1,0].set_title('Gradually ageing mature forests \n (81-200 years)')
ax[1,0].text(0.05, 1.05, 'c', transform=ax[1,0].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = OG_stand_aging_class.plot.imshow(ax=ax[1,1], transform=ccrs.PlateCarree(), cmap='YlGnBu',vmin=0, vmax=100,
                           cbar_kwargs=cbar_kwargs)
ax[1,1].coastlines()
ax[1,1].gridlines()
ax[1,1].set_title('Gradually ageing old-growth forests \n (>200 years)')
ax[1,1].text(0.05, 1.05, 'd', transform=ax[1,1].transAxes,
            fontsize=16, fontweight='bold', va='top')
plt.savefig(os.path.join(plot_dir,'figS10.png'), dpi=300)


