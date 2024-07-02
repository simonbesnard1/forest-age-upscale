#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:24:30 2024

@author: simon
"""

import xarray as xr
from datetime import datetime
import numpy as np

ds = xr.open_dataset('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/GAMIv2-0_2010-2020_class_fraction_0deg50.nc').forest_age

ds = ds.where(ds>0)

# Extract the data for the age class 0-20
data_0_20 = ds.sel(age_class='0-20')

# Create new age classes 0-1, 1-2, ..., 19-20 by dividing the data_0_20
new_data = data_0_20 / 20

# Generate new age class labels
new_age_classes = [f"{i}-{i+1}" for i in range(20)]

# Expand the new data along a new age_class dimension
new_data_expanded = new_data.expand_dims(age_class=new_age_classes, axis=1)

# Stack the new data into the original dataset, replacing the 0-20 age class
# Remove the old 0-20 class
ds = ds.drop_sel(age_class='0-20')

# Combine the new age classes with the existing dataset
final_data = xr.concat([new_data_expanded, ds], dim='age_class').to_dataset()
final_data = final_data.where(np.isfinite(final_data), -9999).astype('float32')

final_data.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
final_data.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
final_data = final_data.rio.write_crs("epsg:4326", inplace=True)

final_data = final_data.assign_attrs(product_name = "Global Age Mapping Integration (GAMI) v2.0",
                                     version = "v2.0", 
                                     institution = "Helmholtz Center Potsdam GFZ German Research Centre for Geosciences",
                                     institute_id = "GFZ-Potsdam",
                                     resolution = '0d50',
                                     _FillValue = -9999,
                                     created_by='Simon Besnard',
                                     contact = 'Simon Besnard (besnard@gfz-potsdam.de) or Nuno Carvalhais (ncarvalhais@bgc-jena.mpg.de)',
                                     references = "https://doi.org/10.5880/GFZ.1.4.2023.006 and https://doi.org/10.5194/essd-13-4881-2021",
                                     frequency = "2010 and 2020",
                                     creation_date =datetime.now().strftime("%d-%m-%Y %H:%M"))

comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in final_data.data_vars}
final_data.to_netcdf('/home/simon/Documents/science/research_paper/age_NEP_ciais/analysis/data/global_input/GAMI_v2/GAMIv2-0_2010-2020_class_fraction_0deg50_31Ageclasses.nc', 
                     encoding=encoding, mode = 'w')
