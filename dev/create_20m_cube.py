#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:57:16 2023

@author: simon
"""
import rioxarray as rio
import numpy as np
import xarray as xr
import glob
import os

items = glob.glob('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/org_data/*_AGB*.tif')
for item in items:
    ds_ = rio.open_rasterio(item)
    ds_ = ds_.rio.reproject("EPSG:4326")
    ds_ = ds_.where(ds_< ds_.attrs["_FillValue"])
    ds_.rio.to_raster("/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/tmp/" + os.path.basename(item),
                           driver="COG")
    
for product in ["_2020_AGB", "_2021_AGB", "_2020_AGB-SD", "_2021_AGB-SD"]:
    files_to_mosaic = glob.glob('/home/besnard/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/tmp/*' + product + ".tif")
    
    files_string = " ".join(files_to_mosaic)
    
    command = "gdal_merge.py -o /home/besnard/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/ESA_CCI" +  product + ".tif -of gtiff " + files_string
    print(os.popen(command).read())

agb_2020 = rio.open_rasterio('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/ESA_CCI_2020_AGB.tif')
agb_2020 = agb_2020.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
agb_2020['time'] = [np.datetime64('2020-01-01')]
agb_2020 = agb_2020.to_dataset(name = 'agb')
agb_2020 = agb_2020.transpose('time', 'latitude', 'longitude')

agb_sd_2020 = rio.open_rasterio('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/ESA_CCI_2020_AGB-SD.tif')
agb_sd_2020 = agb_sd_2020.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
agb_sd_2020['time'] = [np.datetime64('2020-01-01')]
agb_sd_2020 = agb_sd_2020.to_dataset(name = 'agb_sd')
agb_sd_2020 = agb_sd_2020.transpose('time', 'latitude', 'longitude')

agb_2021 = rio.open_rasterio('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/ESA_CCI_2021_AGB.tif')
agb_2021 = agb_2021.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
agb_2021['time'] = [np.datetime64('2021-01-01')]
agb_2021 = agb_2021.transpose('latitude', 'longitude', 'time')
agb_2021 = agb_2021.to_dataset(name = 'agb')

agb_sd_2021 = rio.open_rasterio('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/ESA_CCI_2021_AGB-SD.tif')
agb_sd_2021 = agb_sd_2021.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
agb_sd_2021['time'] = [np.datetime64('2021-01-01')]
agb_sd_2021 = agb_sd_2021.to_dataset(name = 'agb_sd')
agb_sd_2021 = agb_sd_2021.transpose('time', 'latitude', 'longitude')

agb_2020_2021 = xr.concat([agb_2020, agb_2021, agb_sd_2020, agb_sd_2021], dim = 'time')
agb_2020_2021.to_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_20m')
