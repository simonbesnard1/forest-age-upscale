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

#%% Load management data
forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction
management_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ManagementTypeFrac_1deg').where(forest_fraction >0)
intact_forests =  management_fraction['intact_forests'].where(management_fraction['intact_forests'] >0.5)
naturally_regenerated =  management_fraction['naturally_regenerated'].where(management_fraction['naturally_regenerated'] >0.5)



# Plot data
fig, ax = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(11, 4), 
                       constrained_layout=True, gridspec_kw={'wspace': 0, 'hspace': 0})

image = intact_forests.plot.imshow(ax=ax[0], transform=ccrs.PlateCarree(), vmin =.5, vmax =1,
                                            cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                                                               label ='Fraction [adimensional]'))
ax[0].coastlines()
ax[0].gridlines()
ax[0].text(0.05, 1.05, 'a', transform=ax[0].transAxes,
            fontsize=16, fontweight='bold', va='top')
ax[0].set_title('Fraction of unmanaged forests >0.5')


image = naturally_regenerated.plot.imshow(ax=ax[1], transform=ccrs.PlateCarree(),vmin =.5, vmax =1,
                                            cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                                                               label ='Fraction [adimensional]'))

ax[1].coastlines()
ax[1].gridlines()
ax[1].text(0.05, 1.05, 'b', transform=ax[1].transAxes,
            fontsize=16, fontweight='bold', va='top')
ax[1].set_title('Fraction of managed forests >0.5')


# Save the figure
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS19.png', dpi=300)


