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
import scipy.stats as st
import matplotlib as mpl
from scipy import ndimage
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

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

#%% Define functions
def AreaGridlatlon(lats,lons,res_lat,res_lon):
     ER          = 6378160 #Earth radius (m)
     cols        = lons.shape[0]
     londel      = np.abs(res_lon)
     lats1       = lats - res_lat/2.
     lats2       = lats + res_lat/2.
     areavec     = (np.pi/180)*ER**2 * np.abs(np.sin(lats1 * 
                                                     np.pi/180)-np.sin(lats2 * np.pi/180))*londel
     area_grid   = xr.DataArray(np.matmul(areavec[:,np.newaxis],np.ones([1, cols])), 
                               dims=['latitude', 'longitude'],
                               coords={'latitude': lats,
                                       'longitude': lons})
     return(area_grid)
 
def area_weighted_sum(data, res, scalar_area = 1, scalar_mass=1e-15):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res)
    dat_area_weighted = np.nansum(data * area_grid * scalar_area * scalar_mass)
    return dat_area_weighted

def area_weighed_mean(data, res):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res).values
    dat_area_weighted = np.nansum((data * area_grid) / np.nansum(area_grid[~np.isnan(data)]))
    return dat_area_weighted 

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

def violins(data,pos=0,bw_method=None,resolution=50,spread=1,max_num_points=100):
    """violins(data,pos=0,bw_method=None,resolution=50,spread=1)
    Jitter violin plot creater
    Takes points from a distribution and creates data for both a jitter violin and a standard violin plot.
    Parameters
    ----------
    data : numpy array
        The data to build the violin plots from
    pos : float or int
        The position the resulting violin will be centered on
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth. This can be ‘scott’, ‘silverman’, a scalar constant or a callable. If a scalar, this will be used directly as kde.factor. If a callable, it should take a gaussian_kde instance as only parameter and return a scalar. If None (default), ‘scott’ is used. See Notes for more details.
    resolution : int
        The resolution of the resulting violin plot
    spread : int or float
        The spread of the violin plots
     Returns
    -------
    pointx,pointy : numpy arrays
        The resulting data for the jitter violin plot (use with pl.scatter)
    fillx,filly : numpy array
        The resulting data for a standard violin plot (use with pl.fill_between)
    """
    if data.size>max_num_points:
        data = np.random.choice(data,size=max_num_points,replace=False)
    kde    = st.gaussian_kde(data,bw_method=bw_method)
    pointx = data
    pointy = kde.pdf(pointx)
    pointy = pointy/(2*pointy.max())
    fillx  = np.linspace(data.min(),data.max(),resolution)
    filly  = kde.pdf(fillx)
    filly  = filly/(2*filly.max())
    pointy = pos+np.where(np.random.rand(pointx.shape[0])>0.5,-1,1)*np.random.rand(pointx.shape[0])*pointy*spread
    filly  = (pos-filly*spread,pos+filly*spread)
    return(pointx,pointy,fillx,filly)

#%% Load partition age difference
AgeDiff_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiff_1deg').median(dim = 'members')
growing_forest_diff =  AgeDiff_1deg.aging_forest_diff
growing_forest_diff = growing_forest_diff.where(growing_forest_diff>0, 10)
growing_forest_class =  AgeDiff_1deg.aging_forest_class
growing_forest_class = growing_forest_class.where(growing_forest_class > 0)
stand_replaced_diff = AgeDiff_1deg.stand_replaced_diff
stand_replaced_diff = stand_replaced_diff.where(stand_replaced_diff < 0)
stand_replaced_class = AgeDiff_1deg.stand_replaced_class
stand_replaced_class = stand_replaced_class.where(stand_replaced_class >0)

#%% Load lateral flux data
lateral_fluxes_sink = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[["allcropsink", "biofuelcropsink", 'riversink']]
lateral_fluxes_source = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[['allcropsource', "biofuelcropsource", 'lakeriveremis']]
lateral_fluxes_sink_2020 = lateral_fluxes_sink.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')
lateral_fluxes_source_2020 = lateral_fluxes_source.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')

#%% Load NEE data
forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction
land_fraction = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/inversion_jena/s99oc_v2022_daily/NEE.daily.360.180.2020.nc').LF
NEE_RECCAP = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/GCP/GCP2023_inversions_1x1_version1_1_20240124.nc').land_flux_only_fossil_cement_adjusted
NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').median(dim = "ensemble_member") * 1e+15
NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
#NEE_2020 = NEE_2020.where(forest_fraction>0)
NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)
NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
NEE_2020_filtered['latitude'] = forest_fraction['latitude']
NEE_2020_filtered['longitude'] = forest_fraction['longitude']
NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>0)

#%% Load transcom regions
GFED_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/GFED_regions/GFED_regions_360_180_v1.nc').basis_regions
GFED_regions = GFED_regions.where((GFED_regions == 9) | (GFED_regions == 8))
GFED_regions = GFED_regions.where((GFED_regions ==9) | (np.isnan(GFED_regions)), 5)
GFED_regions = GFED_regions.where((GFED_regions ==5) | (np.isnan(GFED_regions)), 6)
GFED_regions = GFED_regions.rename({'lat' : 'latitude', 'lon' : 'longitude'})
transcom_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/transcom_regions/transcom_regions_360_180.nc').transcom_regions
transcom_regions = transcom_regions.reindex(latitude=transcom_regions.latitude[::-1])
transcom_regions = transcom_regions.where(transcom_regions<=11)
transcom_regions = transcom_regions.where((transcom_regions<5) | (transcom_regions>6) )
transcom_regions = transcom_regions.where(np.isfinite(transcom_regions), GFED_regions)
transcom_regions['latitude'] = growing_forest_diff['latitude']
transcom_regions['longitude'] = growing_forest_diff['longitude']

transcom_mask ={"class_7":{"eco_class" : 7, "name": "Eurasia Boreal"},                
                "class_1":{"eco_class":  1, "name": "NA Boreal"},
                "class_8":{"eco_class" : 8, "name": "Eurasia Temperate"},
                "class_11":{"eco_class" : 11, "name": "Europe"},                
                "class_2":{"eco_class" : 2, "name": "NA Temperate"},
                "class_4":{"eco_class" : 4, "name": "SA Temperate"},
                "class_3":{"eco_class" : 3, "name": "SA Tropical"},
                "class_9":{"eco_class" : 9, "name": "Tropical Asia"},
                "class_5":{"eco_class" : 5, "name": "Northern Africa"},
                "class_6":{"eco_class" : 6, "name": "Southern Africa"},
                "class_10":{"eco_class" : 10, "name": "Australia"}}

#%% Load management type
management_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ManagementTypeFrac_1deg').where(forest_fraction >0)
agroforestry = management_fraction.agroforestry
intact_forests = management_fraction.intact_forests
naturally_regenerated = management_fraction.naturally_regenerated
oil_palm = management_fraction.oil_palm
plantation_forest = management_fraction.plantation_forest
planted_forest = management_fraction.planted_forest

#%% Calculate pixel area
EARTH_RADIUS = 6371.0
delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
width_of_longitude = EARTH_RADIUS * delta_lon
delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
height_of_latitude = EARTH_RADIUS * delta_lat
pixel_area = (width_of_longitude * height_of_latitude *
              np.cos(np.deg2rad(planted_forest.latitude))).broadcast_like(planted_forest)

#%% Compute total area per management for each transcom regions.
total_area_agroforestry_class = {}
total_area_intact_forests_class = {}
total_area_naturally_regenerated_class = {}
total_area_oil_palm_class = {}
total_area_planted_forest_class = {}
total_area_plantation_forest_class = {}

for region_ in list(transcom_mask.keys()):
    class_values = transcom_mask[region_]['eco_class']
    class_name = transcom_mask[region_]['name']
    total_area_agroforestry_class[class_name] = (agroforestry.where(transcom_regions==class_values) * pixel_area).sum(dim=['latitude', 'longitude']).values / 10**7 
    total_area_intact_forests_class[class_name] = (intact_forests.where(transcom_regions==class_values) * pixel_area).sum(dim=['latitude', 'longitude']).values / 10**7 
    total_area_naturally_regenerated_class[class_name] = (naturally_regenerated.where(transcom_regions==class_values) * pixel_area).sum(dim=['latitude', 'longitude']).values / 10**7 
    total_area_oil_palm_class[class_name] = (oil_palm.where(transcom_regions==class_values) * pixel_area).sum(dim=['latitude', 'longitude']).values / 10**7 
    total_area_plantation_forest_class[class_name] = (plantation_forest.where(transcom_regions==class_values) * pixel_area).sum(dim=['latitude', 'longitude']).values / 10**7 
    total_area_planted_forest_class[class_name] = (planted_forest.where(transcom_regions==class_values) * pixel_area).sum(dim=['latitude', 'longitude']).values / 10**7 
    

#%% Calculate rgional age pre stand-replacement
AgeDiff_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiff_1deg')

out = []
for member_ in np.arange(20):
    stand_replaced_diff_member = AgeDiff_1deg.sel(members= member_).stand_replaced_diff
    stand_replaced_diff_member = np.abs(stand_replaced_diff_member.where(stand_replaced_diff_member < 0))

    age_stand_replaced_region = {}
    
    for region_ in list(transcom_mask.keys()):
        class_values = transcom_mask[region_]['eco_class']
        class_name = transcom_mask[region_]['name']
        age_stand_replaced_region[class_name] =  area_weighed_mean(stand_replaced_diff_member.where(transcom_regions==class_values), res=1)
        
    # Creating a DataFrame
    df = pd.DataFrame({
        'member': member_,
        'Region': list(age_stand_replaced_region.keys()),
        'age_stand_replacement': list(age_stand_replaced_region.values())
    })
    
    out.append(df)

out_age_pre_replacement= pd.concat(out)
median_out_pre_replacement = out_age_pre_replacement.groupby("Region").median(numeric_only=False)
q5_out_pre_replacement = out_age_pre_replacement.groupby("Region").quantile(numeric_only=False, q=0.05)
q95_out_pre_replacement = out_age_pre_replacement.groupby("Region").quantile(numeric_only=False, q=0.95)

#%% Plot data
cbar_kwargs = dict(orientation='horizontal', shrink=0.7, aspect=40, pad=0.05, spacing='proportional',
                   label ='Fraction [adimensional]')
fig = plt.figure(figsize=(11.5, 9.5), constrained_layout= True)

ax_scatter1 = fig.add_subplot(2, 2, 3)

j =0
for class_ in list(transcom_mask.keys()):
    class_values = transcom_mask[class_]['eco_class']
    class_name = transcom_mask[class_]['name']
    stand_replaced_subset = np.abs(stand_replaced_diff.where(transcom_regions == class_values).values.reshape(-1))
    NEE_2020_subset = NEE_2020.where(transcom_regions == class_values).values.reshape(-1)
    NEE_2020_subset = NEE_2020_subset[np.isfinite(stand_replaced_subset)]
    stand_replaced_subset = stand_replaced_subset[np.isfinite(stand_replaced_subset)]
    IQ_mask = (stand_replaced_subset > np.quantile(stand_replaced_subset, 0.05)) & (stand_replaced_subset < np.quantile(stand_replaced_subset, 0.95))
    stand_replaced_subset = stand_replaced_subset[IQ_mask]
    NEE_2020_subset = NEE_2020_subset[IQ_mask]
    
    # Calculate points for positive and negative values
    #pointx_pos, pointy_pos, _, _ = violins(growing_forest_subset, pos=j, spread=0.3, max_num_points=2000)
    pointx_neg, pointy_neg, _, _ = violins(stand_replaced_subset, pos=j, spread=0.3, max_num_points=500)

    # Plot positive values in red
    #ax_scatter2.scatter(pointy_pos, pointx_pos, color='#7570b3', alpha=0.2, marker='.')
    
    # Plot negative values in blue
    ax_scatter1.scatter(pointy_neg, pointx_neg, color='#d95f02', alpha=0.2, marker='.')

    # Plot the mean as a large diamond
    NEE_region_2010 = np.array(area_weighted_sum(NEE_2020_filtered.where(transcom_regions == class_values), 1))
    
    error_bars_ = np.vstack((median_out_pre_replacement.loc[class_name]['age_stand_replacement'] - q5_out_pre_replacement.loc[class_name]['age_stand_replacement'],
                            q95_out_pre_replacement.loc[class_name]['age_stand_replacement'] - median_out_pre_replacement.loc[class_name]['age_stand_replacement']))
    scatter = ax_scatter1.scatter(j, median_out_pre_replacement.loc[class_name]['age_stand_replacement'], 
                        marker='d', s=200, norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=.05),
                        cmap = 'bwr', c=[NEE_region_2010], edgecolor='black')
    ax_scatter1.errorbar(j, 
                         median_out_pre_replacement.loc[class_name]['age_stand_replacement'], 
                         yerr=error_bars_, 
                         fmt='none', 
                         ecolor='black', 
                         capsize=2)

    
    #ax_scatter2.scatter(j, np.nanquantile(growing_forest_subset, 0.5), marker='d', s=200, color='black', alpha=0.5)
    j +=1

ticks = [-0.6, -0.3, 0, 0.025, 0.05]

plt.colorbar(scatter, ax=ax_scatter1, orientation='horizontal',  ticks=ticks,
             shrink=0.7, aspect=40, pad=0.05, spacing='proportional',
             label='NEE [PgC year$^{-1}$]')

ax_scatter1.set_xticks(np.arange(0,11))
ax_scatter1.set_ylim(0,300)

name_list = [details['name'] for details in transcom_mask.values()]
#ax1.set_xlabel('Age class', size=12)   
ax_scatter1.set_ylabel('Age pre-stand-replacement [years]', size=14)
ax_scatter1.spines['top'].set_visible(False)
ax_scatter1.spines['right'].set_visible(False)
ax_scatter1.text(0.05, 1.1, 'c', transform=ax_scatter1.transAxes,
            fontsize=16, fontweight='bold', va='top')
ax_scatter1.tick_params(labelsize=12)
ax_scatter1.set_xticklabels(name_list, rotation=90, size=12)

ax_scatter2 = fig.add_subplot(2, 2, 4)
x = np.arange(len(total_area_agroforestry_class))  # the label locations
ax_scatter2.bar(x, list(total_area_agroforestry_class.values()), label='Agroforestry', color='#8c510a')
ax_scatter2.bar(x, list(total_area_intact_forests_class.values()), bottom=list(total_area_agroforestry_class.values()), label='Unmanaged forests', color='#d8b365')
bottom_for_naturally_regenerated = np.add(list(total_area_agroforestry_class.values()), list(total_area_intact_forests_class.values()))
ax_scatter2.bar(x, list(total_area_naturally_regenerated_class.values()), bottom=bottom_for_naturally_regenerated, label='Managed forests', color='#f6e8c3')
bottom_for_oil_palm = np.add(bottom_for_naturally_regenerated, list(total_area_naturally_regenerated_class.values()))
ax_scatter2.bar(x, list(total_area_oil_palm_class.values()), bottom=bottom_for_oil_palm, label='Oil palm plantations', color='#c7eae5')
bottom_for_planted = np.add(bottom_for_oil_palm, list(total_area_oil_palm_class.values()))
ax_scatter2.bar(x, list(total_area_planted_forest_class.values()), bottom=bottom_for_planted, label='Planted forests', color='#5ab4ac')
bottom_for_plantation = np.add(bottom_for_planted, list(total_area_planted_forest_class.values()))
ax_scatter2.bar(x, list(total_area_plantation_forest_class.values()), bottom=bottom_for_plantation, label='Plantation forests', color='#01665e')

ax_scatter2.set_xticks(x)
ax_scatter2.set_xticklabels(list(total_area_agroforestry_class.keys()), rotation=90)
ax_scatter2.spines['top'].set_visible(False)
ax_scatter2.spines['right'].set_visible(False)
ax_scatter2.set_ylabel('Area [billion hectares]', size=12)
ax_scatter2.legend(frameon=False, fontsize=10, loc='upper right', bbox_to_anchor=(1, 1.1), ncol=2)
ax_scatter2.text(0.05, 1.1, 'd', transform=ax_scatter2.transAxes, fontsize=16, fontweight='bold', va='top')
ax_scatter2.set_ylim(0, .9)

ax_map1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
image = growing_forest_class.plot.imshow(ax=ax_map1, cmap='YlGnBu', transform=ccrs.PlateCarree(), vmin=0.7,
                           cbar_kwargs=cbar_kwargs)
ax_map1.coastlines()
ax_map1.gridlines()
ax_map1.set_title('Fraction of gradually ageing forests', fontsize=12, pad=12)
ax_map1.text(0.05, 1.1, 'a', transform=ax_map1.transAxes,
            fontsize=16, fontweight='bold', va='top')


ax_map2 = fig.add_subplot(2, 2, 2, projection=ccrs.Robinson())
image = stand_replaced_class.plot.imshow(ax=ax_map2, cmap='YlGnBu', transform=ccrs.PlateCarree(),vmax=0.3,
                            cbar_kwargs=cbar_kwargs)
ax_map2.coastlines()
ax_map2.gridlines()
ax_map2.set_title('Fraction of stand-replaced forests', fontsize=12, pad=12)
ax_map2.text(0.05, 1.1, 'b', transform=ax_map2.transAxes,
            fontsize=16, fontweight='bold', va='top')

# ax_map2 = fig.add_subplot(2, 2, 2, projection=ccrs.Robinson())
# image = stand_replaced_class.plot.imshow(ax=ax_map2, cmap='YlGnBu', vmax=0.3, transform=ccrs.PlateCarree(),
#                                            cbar_kwargs=cbar_kwargs)
# ax_map2.coastlines()
# ax_map2.gridlines()
# ax_map2.set_title('Fraction of younger forests', fontsize=10)
# ax_map2.text(0.05, 1.05, 'c', transform=ax_map2.transAxes,
#             fontsize=16, fontweight='bold', va='top')
#fig.suptitle('General Title for Subplots', fontsize=16)

plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/fig2.png', dpi=300)
