#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:24:01 2024

@author: simon
"""
#%%Load library
import xarray as xr
from datetime import datetime
import numpy as np



#%% Load data
final_data = xr.open_zarr('/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeUpscale_100m') 


#%% Add metadata
final_data["forest_age"] = final_data["forest_age"].assign_attrs(long_name="Forest age using a fusion machine learning and Landsat-based last time since disturbance",
                                                                 units="years",
                                                                 grid_mapping= 'crs', 
                                                                 valid_min= 1.0, 
                                                                 valid_max= 300)
final_data = final_data.where(np.isfinite(final_data), -9999).astype('int16')

final_data.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
final_data.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
final_data = final_data.rio.write_crs("epsg:4326", inplace=True)

final_data = final_data.assign_attrs(product_name = "Global Age Mapping Integration (GAMI) v2.0",
                                     version = "v2.0", 
                                     institution = "Helmholtz Center Potsdam GFZ German Research Centre for Geosciences",
                                     institute_id = "GFZ-Potsdam",
                                     _FillValue = -9999,
                                     created_by='Simon Besnard',
                                     contact = 'Simon Besnard (besnard@gfz-potsdam.de) or Nuno Carvalhais (ncarvalhais@bgc-jena.mpg.de)',
                                     references = "https://doi.org/10.5880/GFZ.1.4.2023.006 and https://doi.org/10.5194/essd-13-4881-2021",
                                     frequency = "2010 and 2020",
                                     creation_date =datetime.now().strftime("%d-%m-%Y %H:%M"))

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in final_data.data_vars}
final_data.to_netcdf('/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0//GAMIv2-0_2010-2020_100m.nc', 
                     encoding=encoding, mode = 'w')


#%% Load data
final_data = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeClass_1deg') 

#%% Add metadata
final_data["forest_age"] = final_data["forest_age"].assign_attrs(long_name="Fraction of each forest age class",
                                                                 units="years",
                                                                 grid_mapping= 'crs', 
                                                                 valid_min= 0, 
                                                                 valid_max= 1)
final_data = final_data.where(np.isfinite(final_data), -9999).astype('float32')

final_data.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
final_data.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
final_data = final_data.rio.write_crs("epsg:4326", inplace=True)

final_data = final_data.assign_attrs(product_name = "Global Age Mapping Integration (GAMI) v2.0",
                                     version = "v2.0", 
                                     institution = "Helmholtz Center Potsdam GFZ German Research Centre for Geosciences",
                                     institute_id = "GFZ-Potsdam",
                                     _FillValue = -9999,
                                     created_by='Simon Besnard',
                                     contact = 'Simon Besnard (besnard@gfz-potsdam.de) or Nuno Carvalhais (ncarvalhais@bgc-jena.mpg.de)',
                                     references = "https://doi.org/10.5880/GFZ.1.4.2023.006 and https://doi.org/10.5194/essd-13-4881-2021",
                                     frequency = "2010 and 2020",
                                     creation_date =datetime.now().strftime("%d-%m-%Y %H:%M"))

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in final_data.data_vars}
final_data.to_netcdf('/project/glm/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0//GAMIv2-0_2010-2020_class_fraction_1deg.nc', 
                     encoding=encoding, mode = 'w')
