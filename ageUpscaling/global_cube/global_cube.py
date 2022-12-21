#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:06:56 2022

@author: simon
"""
from ageUpscaling.core.cube import DataCube

import xarray as xr

import yaml as yml
import os
import glob

class GlobalCube(DataCube):
    
    def __init__(self,
                 base_file_path:str, 
                 cube_config_path:str):
        
        self.base_file_path = base_file_path
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
    
        super().__init__(self.cube_config)

    def generate_cube(self):
        
        if len(glob.glob(os.path.join(self.base_file_path, '*.nc'))) >0 :
            da = xr.open_mfdataset(os.path.join(self.base_file_path, '*.nc'))
            #da = xr.open_dataset(self.base_file_path + '/{var_}.nc'.format(var_= var_name))
        else:
            da = xr.open_dataset(glob.glob(os.path.join(self.base_file_path, '*.nc')))
        if 'lon' in da.coords:
            da = da.rename({'lon': 'longitude'})
            da['longitude'] = self.cube['longitude']
        if 'lat' in da.coords:
            da = da.rename({'lat': 'latitude'})
            da['latitude'] = self.cube['latitude']
            
        for var_name in self.cube_config['output_variables']:
            if var_name not in da.variables:
                raise RuntimeError(f'Failed to create cube: {var_name} is not present in the input dataset')
            
        da = da[self.cube_config['output_variables']].transpose(*self.cube.dims).chunk(chunks=self.cube.chunks)
        
        if isinstance(da, xr.DataArray):
            da = da.to_dataset()
            
        self.update_cube(da)
            
            
            

