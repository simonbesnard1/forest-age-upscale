#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   cube.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for handling the creation and updating of regularized cube zarr files.
"""
import os
from typing import Union

import xarray as xr
from dask.distributed import Client
import dask

from ageUpscaling.core.cube_utils import ComputeCube

class DataCube(ComputeCube):
    """A class for handling the creation and updating of regularized cube zarr files.

    The `DataCube` class inherits from the `ComputeCube` class and adds additional
    functionality for creating and updating data cubes stored in the zarr format.
    The cube is built using the provided coordinates and metadata. The data can be
    written to the cube in parallel using multiple cores.

    Parameters:
    -----------
    cube_config: dict
        A dictionary containing the configuration parameters for the data cube.
        The following keys are required:
        - 'cube_location': str
            Path to the cube .zarr array, which will be created if it does not exist.
        - 'output_writer_params': dict
            A dictionary containing the following keys:
            - 'dims': dict
                A dictionary of coordinate variables. Must have values for all dimensions.
            - 'chunksizes': tuple
                A tuple of chunk sizes in each dimension.
        - 'temporal_resolution': int
            The temporal resolution of the data.
        - 'spatial_resolution': int
            The spatial resolution of the data.
        - 'output_metadata': dict
            A dictionary of metadata for the data cube.
    
    Attributes:
    -----------
    cube_config: dict
        A dictionary containing the configuration parameters for the data cube.
    cube: xr.Dataset
        The data cube stored in an xarray Dataset object.
    """
    
    def __init__(self,
                 cube_config:dict= {}):
        
        super().__init__(cube_config['cube_location'],
                         cube_config['output_writer_params']['dims'],
                         cube_config['output_writer_params']['chunksizes'],
                         cube_config['temporal_resolution'],
                         cube_config['spatial_resolution'],
                         cube_config['output_metadata'])
        
        self.cube_config = cube_config
        
        if not os.path.isdir(self.cube_config['cube_location']):
            self.new_cube()
        self.cube = xr.open_zarr(self.cube_config['cube_location'])
        
    def update_cube(self, 
                     da: Union[xr.DataArray, xr.Dataset],
                     chunks:dict = None,
                     initialize:bool=True) -> None:
        """Update the data cube with the provided xarray Dataset or DataArray.

        This function updates the data cube with the data in the input xarray Dataset or DataArray. 
        If the `initialize` flag is set to `True`, the function will also initialize any new variables 
        found in the input data. If the `chunks` argument is provided, it should be a dictionary with 
        keys representing the names of variables and values representing the chunk sizes for those 
        variables.
        
        Parameters:
        -----------
        da: xr.Dataset or xr.DataArray
            The dataset or data array containing the data to be updated to the cube.
        chunks: dict, optional
            A dictionary specifying the chunk sizes for each variable in the data array. The dictionary 
            should have keys representing variable names and values representing chunk sizes.
        initialize: bool, optional
            Set to False to skip variable initialization. This is faster if the variables are already 
            initialized. Default is True.
        """
        
        if initialize:
            
            if da.__class__ is xr.Dataset:
                self.init_variable(da, njobs= len((set(da.variables) - set(da.coords))))
            elif da.__class__ is xr.DataArray:
                self.init_variable(da)
            
        if chunks is not None:
            #TODO Need to fix this -  
            client = Client(memory_limit= self.cube_config['mem_limit_per_cpu'])
            futures = [self._update(da.sel(latitude = chunk['latitude'], 
                                           longitude = chunk['longitude'])) 
                       for chunk in chunks]
            dask.compute(*futures, num_workers=self.cube_config['njobs'])
            client.close()

        else:
            self._update(da).compute()
         

