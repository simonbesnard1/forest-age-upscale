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
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import os
from ageUpscaling.utils.plotting import filter_nan_gaussian_conserving, calculate_pixel_area, area_weighted_mean, area_weighted_sum, violins

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

#%% Load partition age difference
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction
AgeDiff_1deg =  xr.open_zarr(os.path.join(data_dir,'AgeDiff_1deg')).median(dim = 'members')
growing_forest_diff =  AgeDiff_1deg.aging_forest_diff
growing_forest_diff = growing_forest_diff.where(growing_forest_diff>0, 10)
growing_forest_class =  AgeDiff_1deg.aging_forest_class.where(forest_fraction>.2)
growing_forest_class = growing_forest_class.where(growing_forest_class > 0)
stand_replaced_diff = AgeDiff_1deg.stand_replaced_diff.where(forest_fraction>.2)
stand_replaced_diff = stand_replaced_diff.where(stand_replaced_diff < 0)
stand_replaced_class = AgeDiff_1deg.stand_replaced_class.where(forest_fraction>.2)
stand_replaced_class = stand_replaced_class.where(stand_replaced_class >0)

#%% Load lateral flux data
lateral_fluxes_sink = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[["allcropsink", "biofuelcropsink", 'riversink']]
lateral_fluxes_source = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[['allcropsource', "biofuelcropsource", 'lakeriveremis']]
lateral_fluxes_sink_2020 = lateral_fluxes_sink.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')
lateral_fluxes_source_2020 = lateral_fluxes_source.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')

#%% Load NEE data
land_fraction = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/inversion_jena/s99oc_v2022_daily/NEE.daily.360.180.2020.nc').LF
NEE_RECCAP = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/GCP/GCP2023_inversions_1x1_version1_1_20240124.nc').land_flux_only_fossil_cement_adjusted
NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').median(dim = "ensemble_member") * 1e+15
NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
NEE_2020 = NEE_2020.where(forest_fraction>.2)
NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)
NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
NEE_2020_filtered['latitude'] = forest_fraction['latitude']
NEE_2020_filtered['longitude'] = forest_fraction['longitude']
NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>0.2)

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
management_fraction = xr.open_zarr(os.path.join(data_dir,'ManagementTypeFrac_1deg')).where(forest_fraction >0.2)
agroforestry = management_fraction.agroforestry
intact_forests = management_fraction.intact_forests
naturally_regenerated = management_fraction.naturally_regenerated
oil_palm = management_fraction.oil_palm
plantation_forest = management_fraction.plantation_forest
planted_forest = management_fraction.planted_forest

#%% Calculate pixel area
pixel_area = calculate_pixel_area(planted_forest, 
                                  EARTH_RADIUS = 6378.160, 
                                  resolution=1)

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
AgeDiff_1deg =  xr.open_zarr(os.path.join(data_dir,'AgeDiff_1deg'))

out = []
for member_ in np.arange(20):
    stand_replaced_diff_member = AgeDiff_1deg.sel(members= member_).stand_replaced_diff.where(forest_fraction>.2)
    stand_replaced_diff_member = np.abs(stand_replaced_diff_member.where(stand_replaced_diff_member < 0))

    age_stand_replaced_region = {}
    
    for region_ in list(transcom_mask.keys()):
        class_values = transcom_mask[region_]['eco_class']
        class_name = transcom_mask[region_]['name']
        age_stand_replaced_region[class_name] =  area_weighted_mean(stand_replaced_diff_member.where(transcom_regions==class_values), res=1)
        
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
    pointx_neg, pointy_neg, _, _ = violins(stand_replaced_subset, pos=j, spread=0.3, max_num_points=500)

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
    
    j +=1

ticks = [-0.6, -0.3, 0, 0.025, 0.05]

plt.colorbar(scatter, ax=ax_scatter1, orientation='horizontal',  ticks=ticks,
             shrink=0.7, aspect=40, pad=0.0, spacing='proportional',
             label=r'$\mathrm{NEE\ [PgC\ yr^{-1}]}$')

ax_scatter1.set_xticks(np.arange(0,11))
ax_scatter1.set_ylim(0,300)

name_list = [details['name'] for details in transcom_mask.values()]
ax_scatter1.set_ylabel('Age pre-stand-replacement [years]', size=14)
ax_scatter1.spines['top'].set_visible(False)
ax_scatter1.spines['right'].set_visible(False)
ax_scatter1.text(0.02, 1.12, '(c)', transform=ax_scatter1.transAxes,
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
ax_scatter2.text(0.02, 1.12, '(d)', transform=ax_scatter2.transAxes, fontsize=16, fontweight='bold', va='top')
ax_scatter2.set_ylim(0, .9)

ax_map1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
image = growing_forest_class.plot.imshow(ax=ax_map1, cmap='YlGnBu', transform=ccrs.PlateCarree(), vmin=0.7,
                           cbar_kwargs=cbar_kwargs)
ax_map1.coastlines()
ax_map1.gridlines()
ax_map1.set_title('Fraction of undisturbed ageing forests', fontsize=16, pad=12)
ax_map1.text(0.02, 1.12, '(a)', transform=ax_map1.transAxes,
            fontsize=16, fontweight='bold', va='top')


ax_map2 = fig.add_subplot(2, 2, 2, projection=ccrs.Robinson())
image = stand_replaced_class.plot.imshow(ax=ax_map2, cmap='YlGnBu', transform=ccrs.PlateCarree(),vmax=0.3,
                            cbar_kwargs=cbar_kwargs)
ax_map2.coastlines()
ax_map2.gridlines()
ax_map2.set_title('Fraction of forests replaced by young stands', fontsize=16, pad=12)
ax_map2.text(0.02, 1.12, '(b)', transform=ax_map2.transAxes,
            fontsize=16, fontweight='bold', va='top')
plt.savefig(os.path.join(plot_dir,'fig2.png'), dpi=300)
