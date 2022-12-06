#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:06:56 2022

@author: simon
"""
from ageUpscaling.core.cube import DataCube
import xarray as xr
import glob

class GlobalCube(DataCube):
    
    def __init__(self,
                 base_file_path:str, 
                 cube_location:str, 
                 coords:dict, 
                 chunks:dict):
        
        self.base_file_path = base_file_path
    
        super().__init__(cube_location, coords, chunks)

    def create_cube(self,
                    varnames:dict):
        
        source_files = glob.glob(self.base_file_path + '/**/*.nc', recursive=True)

        for f_ in source_files:
            ds_ = xr.open_dataset(f_)
            self.update_cube(ds_)