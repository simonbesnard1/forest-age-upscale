#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:34:58 2024

@author: simon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:14:30 2024

@author: simon
"""
import xarray as xr
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler

SLURM_NTASKS = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
print(f'Number of jobs requested is {SLURM_NTASKS}')

lat = np.arange(-89.5, 90.5, 2)
lon = np.arange(-179.5, 180.5, 2)
one_degree_grid = xr.DataArray(np.zeros((len(lat), len(lon))), coords=[lat, lon], dims=['lat', 'lon'])
p_values_array = np.full(one_degree_grid.shape, np.nan)

# Load datasets
management_type = xr.open_zarr('/home/besnard/projects/forest-age-upscale/data/cubes/ForestManagement_Lesiv2022_100m').ForestManagementClass.isel(time=0)
forest_age = xr.open_zarr('/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeUpscale_100m').forest_age.isel(members=10)
biomass = xr.open_zarr('/home/besnard/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v4').aboveground_biomass

# Mapping of management types to managed and not managed
management_mapping = {
    11: 0,
    20: 1,
    31: np.nan,
    32: 1,
    40: np.nan,
    53: np.nan
}

def process_pixel(args):
    i, j = args
    lat_min = lat[i] - 1
    lat_max = lat[i] + 1
    lon_min = lon[j] - 1
    lon_max = lon[j] + 1

    subset_biomass = biomass.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).mean(dim = 'time')
    subset_management_type = management_type.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).astype(int)
    subset_forest_age = forest_age.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).mean(dim = 'time')

    subset_management_type, subset_forest_age, subset_biomass = xr.align(subset_management_type, subset_forest_age, subset_biomass, join='inner')
    
    latitudes = subset_management_type.latitude.values
    longitudes = subset_management_type.longitude.values
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    df = pd.DataFrame({
        'management_type': subset_management_type.values.flatten(),
        'forest_age': subset_forest_age.values.flatten(),
        'biomass': subset_biomass.values.flatten(),
        'lat': np.round(np.repeat(latitudes, len(longitudes)), 1),
        'lon': np.round(np.tile(longitudes, len(latitudes)), 1)
    })
    # Apply the management mapping
    df['management_category'] = df['management_type'].map(management_mapping)
    df = df.dropna().reset_index(drop=True)

    if len(df) > 200:
        
        # Create a unique group identifier for each 5-degree by 5-degree window
        df['group'] = df['lat'].astype(str) + '_' + df['lon'].astype(str)
        
        # Ensure groups are unique and consistent
        df['group'] = df['group'].astype('category').cat.codes
        
        df = df.dropna().reset_index(drop=True)
        
        scaler = StandardScaler()
        df[['forest_age', 'biomass']] = scaler.fit_transform(df[['forest_age', 'biomass']])
        
        try:
            management_counts = df['management_category'].value_counts()

            # Extract the counts for managed and not managed categories
            managed_count = management_counts.get(1, 0)  # Assume 1 represents managed
            not_managed_count = management_counts.get(0, 0)  # Assume 0 represents not managed

            if min(managed_count, not_managed_count) / len(df) > 0.1:
                # Mixed-Effects Model
                model = smf.mixedlm("biomass ~ management_category * forest_age", data=df, groups=df["group"])
                result = model.fit(maxiter=200)  # Use 'lbfgs' method and increase max iterations
                
                # Check for convergence
                if not result.converged:
                    interaction_p_value = np.nan
                    
                    print("Warning: Model did not converge.")
                            
                else:
                    interaction_p_value = result.pvalues["management_category:forest_age"]
                return (i, j, interaction_p_value)
            
            else:
                print(f"Unbalanced data at lat: {lat[i]}, lon: {lon[j]}")
                return (i, j, np.nan)
                    
        except (ValueError, KeyError) as e:
            print(f"Error at lat: {lat[i]}, lon: {lon[j]} - {e}")
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
p_value_dataset.to_netcdf('/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/mixed_effect_management.nc', mode='w')

