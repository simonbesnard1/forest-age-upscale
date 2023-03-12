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
import rioxarray as rio

#%% Load data
data_ = xr.open_zarr('/home/besnard/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m')

#%% Here we can create list of chunks for processing the data chunk by chunk. I do not think you can process the data without chunking it because of memory load.
n_chunks = 1000 # Here you can choose the number of chunks you want. It depends on the chunk size you can handle memory wise. The bigger the n_chunks is, the lower your chunk size would be.  
LatChunks = np.array_split(data_.latitude.values, n_chunks)
LonChunks = np.array_split(data_.longitude.values, n_chunks)
chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
               "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
            for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] # You end up with a list of dictionnary where a dictionnary is the latitude and longitude slices of a chunk

#%% Access data of one spatial chunck (You can basically loop over the list of of chunk dictionnaries) 
data_chunk = data_.sel(chunk_dict[0]).transpose('time', 'latitude', 'longitude')
data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}

#%% Write the the to a cloud-optmized geotiff
out_file = "test.tif"
data_chunk.agb.rio.to_raster(raster_path=out_file, driver="COG")

#%% Load back the written cloud optimized geotiff
da = rio.open_rasterio(out_file)
da = da.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
da['time'] = data_['time']


