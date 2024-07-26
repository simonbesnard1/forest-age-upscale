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
import pandas as pd
import numpy as np
import os

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
print(f'Number of jobs requested is {SLURM_NTASKS}')

# Function to sample up to 2000 points per age class
def sample_age_class(df, n=2000):
    sampled_df = df.groupby('age_class').apply(lambda x: x.sample(n=min(len(x), n), random_state=1))
    return sampled_df.reset_index(drop=True)

# Function to process each region in chunks
def process_region(region_, transcom_mask, transcom_regions, biomass, management_type, forest_age):
    class_values = transcom_mask[region_]['eco_class']
    class_name = transcom_mask[region_]['name']
    bounding_box = transcom_regions.where(transcom_regions==class_values, drop=True)
    lat_max = bounding_box.latitude.max().item()
    lat_min = bounding_box.latitude.min().item()
    lon_max = bounding_box.longitude.max().item()
    lon_min = bounding_box.longitude.min().item()    
    subset_biomass = biomass.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    subset_management_type = management_type.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).astype(int)
    subset_forest_age = forest_age.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    
    subset_management_type, subset_forest_age, subset_biomass = xr.align(subset_management_type, subset_forest_age, subset_biomass, join='inner')
    
    df = pd.DataFrame({'region': class_name,
                       'management_type': subset_management_type.values.flatten(),
                       'forest_age': subset_forest_age.values.flatten(),
                       'biomass': subset_biomass.values.flatten()})
    
    # Apply the management mapping
    df['management_category'] = df['management_type'].map(management_mapping)
    df = df.dropna().reset_index(drop=True)
    
    # Create age classes
    age_bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, np.inf]
    age_labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-140',
                  '141-160', '161-180', '181-200', '201-300', '>300']
    df['age_class'] = pd.cut(df['forest_age'], bins=age_bins, labels=age_labels, right=False)
        
    region_df = sample_age_class(df)
    
    del df
    
    return region_df

# Load datasets
management_type = xr.open_zarr('/home/besnard/projects/forest-age-upscale/data/cubes/ForestManagement_Lesiv2022_100m').ForestManagementClass.isel(time=0)
forest_age = xr.open_zarr('/project/glm//scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeUpscale_100m').forest_age.sel(members=1, time = '2020-01-01')
biomass = xr.open_zarr('/home/besnard/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v4_members').aboveground_biomass.sel(members=1, time = '2020-01-01')
forest_fraction = xr.open_zarr('/project/glm//scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction

# Load transcom regions
GFED_regions = xr.open_dataset('/home/besnard/projects/forest-age-upscale/data/global_product/1d00_static/GFED/GFED_regions_360_180_v1.nc').basis_regions
GFED_regions = GFED_regions.where((GFED_regions == 9) | (GFED_regions == 8))
GFED_regions = GFED_regions.where((GFED_regions ==9) | (np.isnan(GFED_regions)), 5)
GFED_regions = GFED_regions.where((GFED_regions ==5) | (np.isnan(GFED_regions)), 6)
GFED_regions = GFED_regions.rename({'lat' : 'latitude', 'lon' : 'longitude'})
transcom_regions = xr.open_dataset('/home/besnard/projects/forest-age-upscale/data/global_product/1d00_static/TRANSCOM/transcom_regions_360_180.nc').transcom_regions
transcom_regions = transcom_regions.reindex(latitude=transcom_regions.latitude[::-1])
transcom_regions = transcom_regions.where(transcom_regions<=11)
transcom_regions = transcom_regions.where((transcom_regions<5) | (transcom_regions>6) )
transcom_regions = transcom_regions.where(np.isfinite(transcom_regions), GFED_regions)
transcom_regions['latitude'] = forest_fraction['latitude']
transcom_regions['longitude'] = forest_fraction['longitude']

transcom_mask ={"class_8":{"eco_class" : 8, "name": "Eurasia Temperate"},
		"class_1":{"eco_class":  1, "name": "NA Boreal"},
                "class_3":{"eco_class" : 3, "name": "SA Tropical"},                
                "class_11":{"eco_class" : 11, "name": "Europe"},                
                "class_4":{"eco_class" : 4, "name": "SA Temperate"},
                "class_9":{"eco_class" : 9, "name": "Tropical Asia"}}

# Mapping of management types to managed and not managed
management_mapping = {
    11: 0,
    20: 1,
    31: np.nan,
    32: 2,
    40: np.nan,
    53: np.nan
}

final_df = []
for region_ in list(transcom_mask.keys()):
    region_df = process_region(region_, transcom_mask, transcom_regions, biomass, management_type, forest_age)
    final_df.append(region_df)

final_df = pd.concat(final_df)
final_df.to_csv('/project/glm//scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/age_biomass_management.csv')

