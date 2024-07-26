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
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
print(f'Number of jobs requested is {SLURM_NTASKS}')

lat = np.arange(-89.5, 90.5, 5)
lon = np.arange(-179.5, 180.5, 5)
one_degree_grid = xr.DataArray(np.zeros((len(lat), len(lon))), coords=[lat, lon], dims=['lat', 'lon'])
p_values_array = np.full(one_degree_grid.shape, np.nan)

# Load datasets
management_type = xr.open_zarr('/home/besnard/projects/forest-age-upscale/data/cubes/ForestManagement_Lesiv2022_100m').ForestManagementClass.isel(time=0)
forest_age = xr.open_zarr('/home/besnard/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/AgeUpscale_100m').isel(time=1).forest_age
biomass = xr.open_zarr('/home/besnard/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v4').isel(time=1).aboveground_biomass

# Mapping of management types to managed and not managed
management_mapping = {
    11: 'Not Managed',
    20: 'Managed',
    31: 'Managed',
    32: 'Managed',
    40: 'Managed',
    53: 'Managed'
}

def process_pixel(args):
    i, j = args
    lat_min = lat[i] - 2.5
   
    lat_max = lat[i] + 2.5
    lon_min = lon[j] - 2.5
    lon_max = lon[j] + 2.5

    subset_biomass = biomass.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    subset_management_type = management_type.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).astype(int)
    subset_forest_age = forest_age.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

    subset_management_type, subset_forest_age, subset_biomass = xr.align(subset_management_type, subset_forest_age, subset_biomass, join='inner')
    
    df = pd.DataFrame({
        'management_type': subset_management_type.values.flatten(),
        'forest_age': subset_forest_age.values.flatten(),
        'biomass': subset_biomass.values.flatten()
    })

    df = df.dropna()

    if len(df) > 10 and 'management_type' in df.columns and 'forest_age' in df.columns and 'biomass' in df.columns:
        # Apply the management mapping
        df['management_category'] = df['management_type'].map(management_mapping)
        
        # Normalize biomass data
        df['biomass'] = (df['biomass'] - df['biomass'].mean()) / df['biomass'].std()

        # Check the balance of managed vs not managed
        managed_count = df['management_category'].value_counts().get('Managed', 0)
        not_managed_count = df['management_category'].value_counts().get('Not Managed', 0)
        
        if min(managed_count, not_managed_count) / len(df) > 0.4:  # Ensure at least 10% of each category
            try:
                model = ols('biomass ~ C(management_category) * forest_age', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=3)
                interaction_p_value = anova_table.loc['C(management_category):forest_age', 'PR(>F)']
                return (i, j, interaction_p_value)
                    
            except (ValueError, KeyError) as e:
                print(f"Error at lat: {lat[i]}, lon: {lon[j]} - {e}")
                return (i, j, np.nan)
        else:
            print(f"Imbalance in managed vs not managed at lat: {lat[i]}, lon: {lon[j]}")
            return (i, j, np.nan)
    else:
        print(f"Insufficient data at lat: {lat[i]}, lon: {lon[j]}")
        return (i, j, np.nan)

with ProcessPoolExecutor(max_workers=SLURM_NTASKS) as executor:
    results = executor.map(process_pixel, [(i, j) for i in range(len(lat)) for j in range(len(lon))])

for i, j, p_value in results:
    p_values_array[i, j] = p_value

p_value_dataset = xr.DataArray(p_values_array, coords=[lat, lon], dims=['latitude', 'longitude'])
p_value_dataset = xr.Dataset({'p_value_interaction': p_value_dataset})
p_value_dataset.to_netcdf('/home/besnard/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/anova_management.nc', mode='w')

