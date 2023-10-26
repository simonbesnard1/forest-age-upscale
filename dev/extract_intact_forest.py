#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:42:46 2023

@author: simon
"""

import geopandas as gpd
from rasterio.features import geometry_mask
import xarray as xr
import numpy as np
import pandas as pd
import simplekml


#%% Load data
intact_forest = gpd.read_file('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/shapefiles/IFL_2020.zip')
intact_tropical_forest = intact_forest[intact_forest['IFL_ID'].str.contains('|'.join(['SAM', 'SEA', 'AFR']))]
subset_agb_cube        = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v4')['aboveground_biomass'].isel(time=1)

#%% Extract intact forest pixels
pixel_coordinates = []

for index, row in intact_tropical_forest.iterrows():
    polygon = row.geometry
    subset_= subset_agb_cube.sel(latitude =slice(polygon.bounds[3], polygon.bounds[1]), longitude =slice(polygon.bounds[0], polygon.bounds[2]))
    polygon_mask = geometry_mask([polygon], out_shape=subset_.shape, transform=subset_.rio.transform())
    
    # Extract the pixel coordinates within the polygon
    lat_indices, lon_indices = np.where(polygon_mask)

    # Convert indices to latitude and longitude coordinates
    latitudes = subset_.latitude[lat_indices]
    longitudes = subset_.longitude[lon_indices]    
    df_ = pd.DataFrame({'latitude': latitudes.values, 'longitude': longitudes.values}).sample(n=500, random_state=42)
    
    pixel_coordinates.append(df_)
    
intact_pixels = pd.concat(pixel_coordinates)
intact_pixels.to_csv('/home/simon/gfz_hpc/projects/forest-age-upscale/data/training_data/intact_forest_data.csv')

#%% Create a KML object
kml = simplekml.Kml()

# Iterate through the DataFrame rows and add placemarks
for index, row in intact_pixels.sample(n=20000, random_state=42).iterrows():
    lat, lon = row['latitude'], row['longitude']
    kml.newpoint(name=str(index), coords=[(lon, lat)])
    
kml.save("/home/simon/gfz_hpc/projects/forest-age-upscale/data/training_data/intact_forest_data.kml")
