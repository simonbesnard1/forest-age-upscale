#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:10:34 2022

@author: simon
"""
from typing import Union
import os

import xarray as xr

from ageUpscaling.core.cube_utils import ComputeCube

class DataCube(ComputeCube):
    """DataCube(cube_location, coords=None, chunks = None, njobs=1)

    Handles creation and updating of regularized cube zarr files.

    Cubes are build with provided coordinates

    Parameters
    ----------
    cube_location : str
        Path to cube .zarr array, which will be created if it does not exist.

    coords : dictionary of coordinates
        `coords` will be passed to xarray.Dataset().

    chunks : dictionary defining chunks
        `chunks` will be passed to xarray.Dataset()

    njobs : int
        Number of cores to use in parallel when writing data.

    """
    def _init_cube(self):
        """
        Reloads the zarr file. If one does not yet exists, an empty one will be created.
        """
        if not os.path.isdir(self.cube_config['cube_location']):
            self.new_cube()
        
        self.cube = xr.open_zarr(self.cube_config['cube_location'])
    
    def __init__(self, 
                 cube_config:dict= {}):
                    
        super().__init__(cube_config)
        
        self._init_cube()
        
    def compute_cube(self, 
                     da: Union[xr.DataArray, xr.Dataset],
                     njobs:int =1,
                     initialize:bool=True):
        """update_cube(da, njobs=None, initialize=True)

        update the cube with the provided Dataset or DataArray.

        Parameters
        ----------
        da : Dataset or DataArray
            should contain the data to be updated to the cube
        njobs : int
            number of CPUs to use in parallel when updating,
            each variable will be updated in parallel
        initialize : bool
            set false to skip variable initialization,
            faster if variables are pre-initialized
        """
        
        if initialize:
            self.init_variable(da, self.cube, njobs=njobs)
            
        self._update(da, njobs=njobs)
    
