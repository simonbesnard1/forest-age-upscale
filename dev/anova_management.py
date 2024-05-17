#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:14:30 2024

@author: simon
"""
import xarray as xr
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

lat = np.arange(-89.5, 90.5, 1)
lon = np.arange(-179.5, 180.5, 1)
one_degree_grid = xr.DataArray(np.zeros((len(lat), len(lon))), coords=[lat, lon], dims=['lat', 'lon'])
p_values_array = np.full(one_degree_grid.shape, np.nan)

# Load datasets
management_type = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/ForestManagement_Lesiv2022_100m').ForestManagementClass.isel(time=0)
forest_age = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/AgeUpscale_100m').isel(time=1).forest_age
biomass = xr.open_zarr('/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v4').isel(time=1).aboveground_biomass

extent_list = []
for i in range(len(lat)):
    for j in range(len(lon)):
        # Define the latitude and longitude bounds for the one-degree pixel
        lat_min = lat[i] - 0.5
        lat_max = lat[i] + 0.5
        lon_min = lon[j] - 0.5
        lon_max = lon[j] + 0.5
    
        subset_biomass = biomass.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
        subset_management_type = management_type.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).astype(int)
        subset_forest_age = forest_age.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

        # Ensure datasets are aligned
        subset_management_type, subset_forest_age, subset_biomass = xr.align(subset_management_type, subset_forest_age, subset_biomass, join='inner')
        
        # Combine datasets into a DataFrame
        df = pd.DataFrame({
            'management_type': subset_management_type.values.flatten(),
            'forest_age': subset_forest_age.values.flatten(),
            'biomass': subset_biomass.values.flatten()
        })
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if len(df) > 0:

            # Fit an OLS model and perform ANOVA
            model = ols('biomass ~ C(management_type) * forest_age', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=3)
            
            interaction_p_value = anova_table.loc['C(management_type):forest_age', 'PR(>F)']
            
            p_values_array[i, j] = interaction_p_value

p_value_dataset = xr.DataArray(p_values_array, coords=[lat, lon], dims=['latitude', 'longitude'])
p_value_dataset = xr.Dataset({'p_value_interaction': p_value_dataset})
p_value_dataset.to_netcdf('/home/simon/hpc_home/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/anova_management.nc', mode= 'w')
