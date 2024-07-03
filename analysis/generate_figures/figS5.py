#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:53:15 2023

@author: simon
"""
#%% load library
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#%% Load data
AgeDiffPartition_fraction_1deg =  xr.open_zarr("/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/AgeDiffPartition_fraction_1deg")
Young_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '0-20')
Young_aging_class = Young_aging_class.where(Young_aging_class >0)

Intermediate_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = ['20-40', '40-60', '60-80']).sum(dim='age_class')
Intermediate_aging_class = Intermediate_aging_class.where(Intermediate_aging_class >0)

Mature_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = ['80-100', '100-120', '120-140', '140-160', '160-180', '180-200']).sum(dim='age_class')
Mature_aging_class = Mature_aging_class.where(Mature_aging_class >0)

OG_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '>200')
OG_aging_class = OG_aging_class.where(OG_aging_class >0)

#%% Plot data
fig, ax = plt.subplots(2,2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(10, 7.5), constrained_layout=True)
cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                   label ='Fraction [adimensional]')

image = Young_aging_class.plot.imshow(ax=ax[0,0], transform=ccrs.PlateCarree(), cmap='YlGnBu', vmin=0, vmax=.3,
                           cbar_kwargs=cbar_kwargs)
ax[0,0].coastlines()
ax[0,0].gridlines()
ax[0,0].set_title('Fraction of gradually ageing young forests \n (1-20 years)')
ax[0,0].text(0.05, 1.05, 'a', transform=ax[0,0].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = Intermediate_aging_class.plot.imshow(ax=ax[0,1], transform=ccrs.PlateCarree(), cmap='YlGnBu', vmin=0, vmax=.3,
                                           cbar_kwargs=cbar_kwargs)
ax[0,1].coastlines()
ax[0,1].gridlines()
ax[0,1].set_title('Fraction of gradually ageing maturing forests \n (21-80 years)')
ax[0,1].text(0.05, 1.05, 'b', transform=ax[0,1].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = Mature_aging_class.plot.imshow(ax=ax[1,0], transform=ccrs.PlateCarree(), cmap='YlGnBu', vmin=0, vmax=.3,
                           cbar_kwargs=cbar_kwargs)
ax[1,0].coastlines()
ax[1,0].gridlines()
ax[1,0].set_title('Fraction of gradually ageing mature forests \n (81-200 years)')
ax[1,0].text(0.05, 1.05, 'c', transform=ax[1,0].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = OG_aging_class.plot.imshow(ax=ax[1,1], transform=ccrs.PlateCarree(), cmap='YlGnBu',vmin=0, vmax=.3,
                           cbar_kwargs=cbar_kwargs)
ax[1,1].coastlines()
ax[1,1].gridlines()
ax[1,1].set_title('Fraction of gradually ageing old-growth forests \n (>200 years)')
ax[1,1].text(0.05, 1.05, 'd', transform=ax[1,1].transAxes,
            fontsize=16, fontweight='bold', va='top')
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS5.png', dpi=300)


