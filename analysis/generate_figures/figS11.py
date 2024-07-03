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
import numpy as np
from scipy import ndimage

def filter_nan_gaussian_conserving(ds_: xr.DataArray, 
                                   length_km: float = 1000, 
                                   length_degree_longitude_equator_km: float = 112.32) -> xr.DataArray:
    """
    Apply a Gaussian filter to an xarray DataArray, preserving the total intensity 
    while considering NaN values.

    This function applies a Gaussian filter to the input DataArray `ds_`. The 
    filtering conserves the total 'intensity' (sum of the values) by redistributing
    intensity only among non-NaN pixels. The NaN values in the input DataArray remain
    NaN in the output. The Gaussian distribution weights used for intensity 
    redistribution consider only available (non-NaN) pixels. 

    The smoothing scale of the Gaussian filter is determined by `length_km`, which 
    is the physical length scale in kilometers. The sigma of the Gaussian filter is 
    calculated based on the length in degrees of longitude at the equator, given by
    `length_degree_longitude_equator_km`.

    Parameters:
    ds_ (xr.DataArray): The input DataArray to be filtered. It should contain NaN 
                        values to indicate missing data.
    length_km (float, optional): The physical length scale in kilometers for the 
                                 Gaussian filter. Default is 1000 km.
    length_degree_longitude_equator_km (float, optional): The length in degrees of 
                                                          longitude at the equator, 
                                                          used for calculating sigma 
                                                          of the Gaussian filter. 
                                                          Default is 112.32 km.

    Returns:
    xr.DataArray: A new DataArray that has been smoothed with a Gaussian filter. 
                  NaN values from the original DataArray are preserved.
    """

    sigma = length_km / length_degree_longitude_equator_km 
    
    arr = ds_.values
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr
    
    out_ = xr.DataArray(gauss, coords=ds_.coords)

    return out_

#%% Load forest fraction
forest_fraction = forest_fraction = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/ForestFraction_1deg').forest_fraction

#%% Load lateral flux data
lateral_fluxes_sink = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[["allcropsink", "biofuelcropsink", 'riversink']]
lateral_fluxes_source = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[['allcropsource', "biofuelcropsource", 'lakeriveremis']]
lateral_fluxes_sink_2010 = lateral_fluxes_sink.sel(time= slice('2009-01-01', '2011-12-31')).mean(dim='time').to_array().sum(dim='variable')
lateral_fluxes_sink_2020 = lateral_fluxes_sink.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')
lateral_fluxes_source_2010 = lateral_fluxes_source.sel(time= slice('2009-01-01', '2011-12-31')).mean(dim='time').to_array().sum(dim='variable')
lateral_fluxes_source_2020 = lateral_fluxes_source.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')

#%% Load NEE data
land_fraction = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/inversion_jena/s99oc_v2022_daily/NEE.daily.360.180.2020.nc').LF
NEE_2010 =  xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/GCP/GCP2022_inversions_1x1_version1_2_20230428.nc').land_flux_only_fossil_cement_adjusted.sel(time= slice('2009-01-01', '2011-12-31')).median(dim = "ensemble_member").mean(dim='time') * 1e+15
NEE_2010 = NEE_2010.where(land_fraction>0)
NEE_2010 = NEE_2010.reindex(latitude=NEE_2010.latitude[::-1])
NEE_2010['latitude'] = forest_fraction['latitude']
NEE_2010['longitude'] = forest_fraction['longitude']
NEE_2010 = NEE_2010 - lateral_fluxes_sink_2010 + lateral_fluxes_source_2010
NEE_2010_filtered = filter_nan_gaussian_conserving(NEE_2010)
NEE_2010_filtered = NEE_2010_filtered.where(forest_fraction>0)
NEE_2010 = NEE_2010.where(forest_fraction>0)

#%% Plot data
fig, ax = plt.subplots(2,2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(10, 7.5), constrained_layout=True)
cbar_kwargs = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional',
                   label ='NEE [gC m$^{-2}$ year$^{-1}$]')

image = NEE_2010.plot.imshow(ax=ax[0,0], transform=ccrs.PlateCarree(), cmap='bwr', vmin=-200, vmax=200,
                           cbar_kwargs=cbar_kwargs)
ax[0,0].coastlines()
ax[0,0].gridlines()
ax[0,0].set_title('NEE 2010 non-filtered')
ax[0,0].text(0.05, 1.05, 'a', transform=ax[0,0].transAxes,
            fontsize=16, fontweight='bold', va='top')

image = NEE_2010_filtered.plot.imshow(ax=ax[0,1], transform=ccrs.PlateCarree(), cmap='bwr', vmin=-200, vmax=200,
                                           cbar_kwargs=cbar_kwargs)
ax[0,1].coastlines()
ax[0,1].gridlines()
ax[0,1].set_title('NEE 2010 smoothed using Gaussian filter')
ax[0,1].text(0.05, 1.05, 'b', transform=ax[0,1].transAxes,
            fontsize=16, fontweight='bold', va='top')

diff_ = NEE_2010 - NEE_2010_filtered

image = diff_.plot.imshow(ax=ax[1,1], transform=ccrs.PlateCarree(), cmap='bwr',vmin=-100, vmax=100,
                           cbar_kwargs=cbar_kwargs)
ax[1,1].coastlines()
ax[1,1].gridlines()
ax[1,1].set_title('NEE 2010 non-filtered - NEE 2010 smoothed')
ax[1,1].text(0.05, 1.05, 'c', transform=ax[1,1].transAxes,
            fontsize=16, fontweight='bold', va='top')
fig.delaxes(ax[1,0])

plt.savefig('/home/simon/Desktop/smooth_NEE_map.png', dpi=300)


