#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   global_cube.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for creating global cube
"""
import xarray as xr
import yaml as yml
import os
import glob

from ageUpscaling.core.cube import DataCube

class GlobalCube(DataCube):
    """GlobalCube is a subclass of DataCube that is used to create a global datacube from a base file and a cube configuration file.
    
    Parameters:
        base_file_path: str
            Path to the base file.
        cube_config_path: str
            Path to the cube configuration file.
    """
    def __init__(self,
                 base_file_path:str, 
                 cube_config_path:str):
        
        self.base_file_path = base_file_path
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
    
        super().__init__(self.cube_config)

    def generate_cube(self) -> None:
        """Generate a data cube from input datasets.
   
        Parameters:
            base_file_path: str
                Path to the input dataset.
            cube_config_path: str
                Path to the configuration file for the data cube.
        """
        for file_ in  glob.glob(os.path.join(self.base_file_path, '*.nc')):
            da = xr.open_dataset(file_)
            if 'lon' in da.coords:
                da = da.rename({'lon': 'longitude'})
                da['longitude'] = self.cube['longitude']
            if 'lat' in da.coords:
                da = da.rename({'lat': 'latitude'})
                da['latitude'] = self.cube['latitude']
            
            vars_to_proc = {}
            for var_name in self.cube_config['output_variables']:
                if var_name in da.variables:
                    vars_to_proc[var_name] = var_name
                    
            if len(vars_to_proc) > 0:        
                da = da[list(vars_to_proc)].transpose(*self.cube.dims).chunk(chunks=self.cube.chunks)
                
                self.update_cube(da)
                    
            
            

