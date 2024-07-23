#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File    :   BacciniBiomasss.py

This module provides functionalities for handling the creation and updating of regularized cube zarr files.

Example usage:
--------------
from BacciniBiomasss import BacciniBiomasss

# Create a Baccini biomass data cube
base_file = 'path/to/base_file'
cube_config_path = 'path/to/cube_config.yml'
biomass_cube = BacciniBiomasss(base_file, cube_config_path)
biomass_cube.CreateCube(var_name='aboveground_biomass', chunk_data=True, n_workers=10)
"""

from ageUpscaling.core.cube import DataCube
import yaml as yml
import rioxarray as rio 
import numpy as np
from itertools import product
import pandas as pd
import os

class BacciniBiomasss(DataCube):
    """
    BacciniBiomasss is a subclass of DataCube that is used to create an aboveground biomass data cube 
    from a base file and a cube configuration file.

    Parameters
    ----------
    base_file : str
        Path to the base file.
    cube_config_path : str
        Path to the cube configuration file.
    """

    def __init__(self,
                 base_file:str, 
                 cube_config_path:str):
        
        self.base_file = base_file
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)        
        
        self.da =  rio.open_rasterio(base_file).rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'})
        self.da['time'] = [pd.to_datetime(os.path.basename(base_file).split('_100m')[0].split('baccini')[1] + '-01-01')]
        
        self.cube_config['output_writer_params']['dims']['latitude'] = self.da.latitude.values
        self.cube_config['output_writer_params']['dims']['longitude'] = self.da.longitude.values
        self.cube_config['output_metadata']['scale_factor'] = self.da.scale_factor
        self.cube_config['output_metadata']['add_offset'] = self.da.add_offset
        self.cube_config['output_metadata']['_FillValue'] = self.da._FillValue
        
        super().__init__(self.cube_config)

    def CreateCube(self, 
                  var_name:str= 'aboveground_biomass',
                  chunk_data:bool = False,
                  n_workers:int=10) -> None:
        """
        Fill a data cube from input datasets.

        This function processes input datasets stored at the path specified in the `base_file` attribute,
        and generates a data cube based on the configuration specified in the `cube_config` attribute.

        The function will only include variables specified in the 'output_variables' field of the `cube_config` dictionary.
        The resulting data array will be transposed to the dimensions specified in the `dims` attribute of the
        `cube` attribute. The data array is then split into chunks and processed by separate workers using the
        Dask library, with the number of workers specified.

        Parameters
        ----------
        var_name : str, optional
            The name of the variable to be included in the data cube. Default is 'aboveground_biomass'.
        chunk_data : bool, optional
            Whether to split the data into chunks for parallel processing. Default is False.
        n_workers : int, optional
            The number of workers to use for parallel processing. Default is 10.

        Raises
        ------
        ValueError
            If the variable specified in `var_name` is not in the data cube configuration.
        """
        
        ds_ = self.da.to_dataset(name = var_name).transpose('latitude', 'longitude', 'time')
        
        vars_to_proc = {}
        
        for var_ in self.cube_config['cube_variables']:
            
            if var_ in ds_.variables:
                vars_to_proc[var_] = self.cube_config['cube_variables'][var_]
        
        if len(vars_to_proc) > 0:
            
            self.init_variable(vars_to_proc, 
                               njobs= len(vars_to_proc))
            
            ds_ = ds_[vars_to_proc.keys()].transpose(*self.cube.dims)
            
            if chunk_data:
            
                LatChunks = np.array_split(ds_.latitude.values, self.cube_config["num_chunks"])
                LonChunks = np.array_split(ds_.longitude.values, self.cube_config["num_chunks"])
                
                to_proc = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                            "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                            for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
                
            else:
                to_proc= None
                
            self.CubeWriter(ds_, chunks=to_proc, n_workers=n_workers)
            
        else: 
            raise ValueError(f'{var_name} is not in the data cube configuration file define in the cube_config_path parameters')
                
                    
