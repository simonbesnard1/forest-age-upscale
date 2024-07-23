#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File    :   SoilGrids.py

This module provides functionalities for handling the creation and updating of regularized cube zarr files.

Example usage:
--------------
from SoilGrids import SoilGrids

# Create a SoilGrids data cube
base_file = 'path/to/base_file'
cube_config_path = 'path/to/cube_config.yml'
soil_cube = SoilGrids(base_file, cube_config_path)
soil_cube.CreateCube(var_name='bdod_0_5cm_mean', chunk_data=True, n_workers=10)
"""
from ageUpscaling.core.cube import DataCube
import rioxarray as rio 
import yaml as yml
import os
import xarray as xr
import numpy as np
from itertools import product

class SoilGrids(DataCube):
    """
    SoilGrids is a subclass of DataCube that is used to create a SoilGrids data cube 
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
        
        if (os.path.basename(self.base_file).split('.')[-1] == 'tif'):    
            self.da =  rio.open_rasterio(base_file)     
            
        else:
            self.da = xr.open_dataset(self.base_file)
        
        self.da =  self.da.sel(band=1).rename({'x': 'longitude', 'y': 'latitude'})
               
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
            
        self.cube_config['output_writer_params']['dims']['latitude'] = self.da.latitude.values
        self.cube_config['output_writer_params']['dims']['longitude'] = self.da.longitude.values
        self.cube_config['output_metadata']['scale_factor'] = self.da.scale_factor
        self.cube_config['output_metadata']['add_offset'] = self.da.add_offset
        self.cube_config['output_metadata']['_FillValue'] = self.da._FillValue
    
        super().__init__(self.cube_config)

    def CreateCube(self, 
                  var_name:str= 'bdod_0_5cm_mean',
                  chunk_data:bool = False,
                  n_workers:int=10) -> None:
        """
        Fill a data cube from input datasets.

        This function processes input datasets stored at the path specified in the `base_file` attribute,
        and generates a data cube based on the configuration specified in the `cube_config` attribute.

        The function will rename longitude and latitude coordinates to 'longitude' and 'latitude', respectively,
        and only include variables specified in the 'output_variables' field of the `cube_config` dictionary.
        The resulting data array will be transposed to the dimensions specified in the `dims` attribute of the
        `cube` attribute. The data array is then split into chunks and processed by separate workers using the
        Dask library, with the number of workers specified.

        Parameters
        ----------
        var_name : str, optional
            The name of the variable to be included in the data cube. Default is 'bdod_0_5cm_mean'.
        chunk_data : bool, optional
            Whether to split the data into chunks for parallel processing. Default is False.
        n_workers : int, optional
            The number of workers to use for parallel processing. Default is 10.

        Raises
        ------
        ValueError
            If the variable specified in `var_name` is not in the data cube configuration.
        """
        
        ds_ = self.da.to_dataset(name = var_name)
        
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
                
                    
