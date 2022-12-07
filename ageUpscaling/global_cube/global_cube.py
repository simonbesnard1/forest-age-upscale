#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:06:56 2022

@author: simon
"""
from ageUpscaling.core.cube import DataCube

import xarray as xr

import yaml as yml

class GlobalCube(DataCube):
    
    def __init__(self,
                 base_file_path:str, 
                 cube_config_path:str):
        
        self.base_file_path = base_file_path
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
    
        super().__init__(self.cube_config)

    def generate_cube(self):
                
        for var_name in self.cube_config['output_variables']:
            da = xr.open_dataset(self.base_file_path + '/{var_}.nc'.format(var_= var_name))
            self.compute_cube(da)
            