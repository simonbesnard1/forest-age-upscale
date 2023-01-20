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
import numpy as np
from itertools import product

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

        This function processes input datasets stored at the path specified in the `base_file_path` attribute, 
        and generates a data cube based on the configuration specified in the `cube_config` attribute.
    
        The function will rename longitude and latitude coordinates to 'longitude' and 'latitude', respectively, 
        and only include variables specified in the 'output_variables' field of the `cube_config` dictionary. 
        The resulting data array will be transposed to the dimensions specified in the `dims` attribute of the 
        `cube` attribute. The data array is then split into chunks and processed by separate workers using the 
        Dask library, with the number of workers specified.
    
        """
        for f_ in glob.glob(os.path.join(self.base_file_path, '*.nc')):
            da = xr.open_dataset(f_)
                
            if 'lon' in da.coords:
                da = da.rename({'lon': 'longitude'})
                da['longitude'] = self.cube['longitude']
            if 'lat' in da.coords:
                da = da.rename({'lat': 'latitude'})
                da['latitude'] = self.cube['latitude']
            
            vars_to_proc = {}
            for var_name in self.cube_config['cube_variables']:
                if var_name in da.variables:
                    vars_to_proc[var_name] = var_name
               
            if len(vars_to_proc) > 0:        
                da = da[vars_to_proc.keys()].transpose(*self.cube.dims)
                LatChunks = np.array_split(da.latitude.values, self.cube_config["num_chunks"])
                LonChunks = np.array_split(da.longitude.values, self.cube_config["num_chunks"])
                
                to_proc = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                            "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                           for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
                
                self.update_cube(da, chunks=to_proc)
                
                    
            
            

