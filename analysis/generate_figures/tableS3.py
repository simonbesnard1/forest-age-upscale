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
import numpy as np
import pandas as pd
import os
from ageUpscaling.utils.plotting import calculate_pixel_area

#%% Specify data and plot directories
data_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/data/Age_upscale_100m'

#%% Load forest fraction
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction

#%% Load partition age difference
AgeDiff_1deg =  xr.open_zarr(os.path.join(data_dir,'AgeDiff_1deg'))
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction
pixel_area = calculate_pixel_area(forest_fraction, 
                                  EARTH_RADIUS = 6378.160, 
                                  resolution=1)
out = []
for member_ in np.arange(20):
    
    growing_forest_diff =  AgeDiff_1deg.sel(members= member_).aging_forest_diff.where(forest_fraction >0.2)
    growing_forest_diff = growing_forest_diff.where(growing_forest_diff>0, 10)
    growing_forest_class =  AgeDiff_1deg.sel(members= member_).aging_forest_class.where(forest_fraction >0.2)
    stand_replaced_diff = AgeDiff_1deg.sel(members= member_).stand_replaced_diff.where(forest_fraction >0.2)
    stand_replaced_class = AgeDiff_1deg.sel(members= member_).stand_replaced_class.where(forest_fraction >0.2)
    
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
    
    #%% Compute total area per management for each transcom regions.
    growing_forest_class = growing_forest_class.where(np.isfinite(transcom_regions))
    stand_replaced_class = stand_replaced_class.where(np.isfinite(transcom_regions))
    total_area_ageing = (growing_forest_class * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_stand_replaced = (stand_replaced_class * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    
    total_area_ageing_region = {}
    total_area_stand_replaced_forests_region = {}
    ratio_area = {}
    
    for region_ in list(transcom_mask.keys()):
        class_values = transcom_mask[region_]['eco_class']
        class_name = transcom_mask[region_]['name']
        total_area_ageing_region[class_name] = (growing_forest_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
        total_area_stand_replaced_forests_region[class_name] = (stand_replaced_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
        ratio_area[class_name] = total_area_stand_replaced_forests_region[class_name] / total_area_ageing_region[class_name]
        
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
        'area_stand_replaced': list(total_area_stand_replaced_forests_region.values()),
        'fraction_stand_replaced': list(total_area_stand_replaced_forests_region.values()) / total_area_stand_replaced,
    })
    
    out.append(df)

out= pd.concat(out)
median_out = out.groupby("Region").median(numeric_only=True)
q5_out = out.groupby("Region").quantile(numeric_only=True, q=0.05)
q95_out = out.groupby("Region").quantile(numeric_only=True, q=0.95)

    