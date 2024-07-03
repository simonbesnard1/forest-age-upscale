#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:22:16 2023

@author: simon
"""
import xarray as xr
import numpy as np
import pandas as pd


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
 
def area_weighed_mean(data, res):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res).values
    dat_area_weighted = np.nansum((data * area_grid) / np.nansum(area_grid[~np.isnan(data)]))
    return dat_area_weighted 

#%% Load transcom regions
GFED_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/GFED_regions/GFED_regions_360_180_v1.nc').basis_regions
#GFED_regions = GFED_regions.where((GFED_regions == 9) | (GFED_regions == 8))
#GFED_regions = GFED_regions.where((GFED_regions ==9) | (np.isnan(GFED_regions)), 5)
#GFED_regions = GFED_regions.where((GFED_regions ==5) | (np.isnan(GFED_regions)), 6)
GFED_regions = GFED_regions.rename({'lat' : 'latitude', 'lon' : 'longitude'})
# GFED_regions['latitude'] = Young_stand_replaced_class['latitude']
# GFED_regions['longitude'] = Young_stand_replaced_class['longitude']
transcom_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/transcom_regions/transcom_regions_360_180.nc').transcom_regions
transcom_regions = transcom_regions.reindex(latitude=transcom_regions.latitude[::-1])
transcom_regions = transcom_regions.where(transcom_regions<=11)

#transcom_regions = transcom_regions.where(transcom_regions<=11)
#transcom_regions = transcom_regions.where((transcom_regions<5) | (transcom_regions>6) )
#transcom_regions = transcom_regions.where(np.isfinite(transcom_regions), GFED_regions)
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

#%% Load partition age difference
out = []
for member_ in np.arange(20):
    
    average_age = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestAge_1deg').forest_age.sel(members = member_)
    average_age_2010 = average_age.sel(time = '2010-01-01')
    average_age_2020 = average_age.sel(time = '2020-01-01')

    age2010_region = {}
    age2020_region = {}
    for region_ in list(transcom_mask.keys()):
        class_values = transcom_mask[region_]['eco_class']
        class_name = transcom_mask[region_]['name']
        age2010_region[class_name] = area_weighed_mean(average_age_2010.where(transcom_regions==class_values), res=1)
        age2020_region[class_name] = area_weighed_mean(average_age_2020.where(transcom_regions==class_values), res=1)
        
    # Creating a DataFrame
    df = pd.DataFrame({
        'member': member_,
        'Region': list(age2010_region.keys()),
        'age_2010': list(age2010_region.values()),
        'age_2020': list(age2020_region.values()),
    })
    df['diff_age'] = df['age_2020'] - df['age_2010']
    df['rel_diff_age'] = (df['diff_age'] / df['age_2010']) *100
    
    out.append(df)

out= pd.concat(out)
median_out = out.groupby("Region").median(numeric_only=True)
q5_out = out.groupby("Region").quantile(numeric_only=True, q=0.05)
q95_out = out.groupby("Region").quantile(numeric_only=True, q=0.95)

