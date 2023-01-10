#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 09:57:16 2023

@author: simon
"""
import rioxarray as rio
import numpy as np
import xarray as xr

agb_2020 = rio.open_rasterio('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/ESA_CCI_2020_AGB.tif')
agb_2020 = agb_2020.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
agb_2020['time'] = [np.datetime64('2020-01-01')]
agb_2020 = agb_2020.transpose('latitude', 'longitude', 'time')
agb_2020 = agb_2020.where(agb_2020>0)
agb_2020 = agb_2020.to_dataset(name = 'agb')

agb_2021 = rio.open_rasterio('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/ESA_CCI_2021_AGB.tif')
agb_2021 = agb_2021.rename({'band': 'time', 'x': 'longitude', 'y': 'latitude'})
agb_2021['time'] = [np.datetime64('2021-01-01')]
agb_2021 = agb_2021.transpose('latitude', 'longitude', 'time')
agb_2021 = agb_2021.where(agb_2021>0)
agb_2021 = agb_2021.to_dataset(name = 'agb')

agb_2020_2021 = xr.concat([agb_2020, agb_2021], dim = 'time')
agb_2020_2021 = agb_2020_2021.rio.reproject("EPSG:4326")
agb_2020_2021.to_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_20m')
