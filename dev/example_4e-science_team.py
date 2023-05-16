#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:51:39 2023

@author: simon
"""
#%% Load library
import xarray as xr
import numpy as np
from itertools import product

#%% Load data
data_ = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeUpscale_100m').sel(members =0)

#%% Create list of chunks
n_chunks = 40 # Here you can choose the number of chunks you want. It depends on the chunk size you can handle memory wise. The bigger the n_chunks is, the lower your chunk size would be.  
LatChunks = np.array_split(data_.latitude.values, n_chunks)
LonChunks = np.array_split(data_.longitude.values, n_chunks)
chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
               "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
            for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] # You end up with a list of dictionnary where a dictionnary is the latitude and longitude slices of a chunk

for chunck in chunk_dict:
    
    for year_ in ['2010', '2018']:    
        data_chunk = data_.sel(chunck).sel(time = year_ + '-01-01').transpose('latitude', 'longitude').forest_age_TC000
        data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
        data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
        data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
        data_chunk.attrs = {'long_name': 'Forest age with no tree cover correction',
                            'units': 'years',
                            'valid_max': 300,
                            'valid_min': 0.0}
        
        # Write the the to a cloud-optmized geotiff
        out_dir = '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/example_output/'
        out_file =  out_dir +  "ForestAge" + year_ +"_" + str(np.round(np.median(data_chunk.latitude), 4)) + '_' + str(np.round(np.median(data_chunk.longitude), 4)) + '.tiff'
        data_chunk.rio.to_raster(raster_path=out_file, driver="COG")

# #%% Load back the written cloud optimized geotiff
# da = rio.open_rasterio(out_file)
# da = da.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
# da['time'] = data_['time']


