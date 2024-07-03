#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:22:16 2023

@author: simon
"""
import xarray as xr
import numpy as np
import pandas as pd

#%% Load partition age difference
AgeDiff_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiff_1deg')

out = []
for member_ in np.arange(20):
    forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction
    
    growing_forest_diff =  AgeDiff_1deg.sel(members= member_).aging_forest_diff
    growing_forest_diff = growing_forest_diff.where(growing_forest_diff>0, 10)
    growing_forest_class =  AgeDiff_1deg.sel(members= member_).aging_forest_class
    #growing_forest_class = growing_forest_class.where(growing_forest_class > 0)
    stand_replaced_diff = AgeDiff_1deg.sel(members= member_).stand_replaced_diff
    stand_replaced_class = AgeDiff_1deg.sel(members= member_).stand_replaced_class
    #stand_replaced_class = stand_replaced_class.where(stand_replaced_class >0)
    #carbon_density_ageing = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/BiomassPartition_1deg_v1').mean(dim = 'age_class').gradually_ageing
    #carbon_density_stand_replacement = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/BiomassPartition_1deg_v1').mean(dim = 'age_class').stand_replaced
    
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
    
    #%% Calculate pixel area
    EARTH_RADIUS = 6371.0
    delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    width_of_longitude = EARTH_RADIUS * delta_lon
    delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    height_of_latitude = EARTH_RADIUS * delta_lat
    pixel_area = (width_of_longitude * height_of_latitude *
                  np.cos(np.deg2rad(stand_replaced_class.latitude))).broadcast_like(stand_replaced_class)
    
    #%% Compute total area per management for each transcom regions.
    growing_forest_class = growing_forest_class.where(np.isfinite(transcom_regions))
    stand_replaced_class = stand_replaced_class.where(np.isfinite(transcom_regions))
    total_area_ageing = (growing_forest_class * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_stand_replaced = (stand_replaced_class * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    
    total_area_ageing_region = {}
    total_area_stand_replaced_forests_region = {}
    ratio_area = {}
    #carbon_stocks_ageing = {}
    #carbon_stocks_stand_replaced = {}
    
    for region_ in list(transcom_mask.keys()):
        class_values = transcom_mask[region_]['eco_class']
        class_name = transcom_mask[region_]['name']
        total_area_ageing_region[class_name] = (growing_forest_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
        total_area_stand_replaced_forests_region[class_name] = (stand_replaced_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
        ratio_area[class_name] = total_area_stand_replaced_forests_region[class_name] / total_area_ageing_region[class_name]
        #carbon_stocks_ageing[class_name] = (growing_forest_class.where(transcom_regions==class_values) * pixel_area * carbon_density_ageing).sum(dim=['latitude', 'longitude']).values * 1e-13 * 0.5
        #carbon_stocks_stand_replaced[class_name] = (stand_replaced_class.where(transcom_regions==class_values) * pixel_area * carbon_density_stand_replacement).sum(dim=['latitude', 'longitude']).values * 1e-13 * 0.5
    total_area_ageing_region['global'] = total_area_ageing
    total_area_stand_replaced_forests_region['global'] = total_area_stand_replaced
    ratio_area['global'] = total_area_stand_replaced / total_area_ageing
    # Creating a DataFrame
    df = pd.DataFrame({
        'member': member_,
        'Region': list(total_area_ageing_region.keys()),
        'area_aging': list(total_area_ageing_region.values()),
        'fraction_aging': list(total_area_ageing_region.values()) / total_area_ageing,
        'ratio_area': list(ratio_area.values()),        
        #'Gradually Aging Carbon Stocks (PgC)': list(carbon_stocks_ageing.values()),
        'area_stand_replaced': list(total_area_stand_replaced_forests_region.values()),
        'fraction_stand_replaced': list(total_area_stand_replaced_forests_region.values()) / total_area_stand_replaced,
        #'Stand-Replaced Carbon Stocks (PgC)': list(carbon_stocks_stand_replaced.values())
    })
    
    out.append(df)

out= pd.concat(out)
median_out = out.groupby("Region").median(numeric_only=True)
q5_out = out.groupby("Region").quantile(numeric_only=True, q=0.05)
q95_out = out.groupby("Region").quantile(numeric_only=True, q=0.95)

    