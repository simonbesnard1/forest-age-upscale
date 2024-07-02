#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   cube_utils.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Code adapted from Jake Nelson cube_utils modules
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam

This module provides functionalities for creating new cube datasets.

Example usage:
--------------
from cube_utils import ComputeCube

cube = ComputeCube(
    cube_location='path/to/cube',
    dims={'time': 365, 'latitude': 180, 'longitude': 360},
    chunksizes=(1, 180, 360),
    temporal_resolution=86400,  # one day in seconds
    spatial_resolution=1000,    # 1 km
    output_metadata={'description': 'Sample cube dataset'}
)
"""

from abc import ABC
from typing import Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc

class ComputeCube(ABC):
    """
    A schema for creating new cube datasets.

    This class can be used to create new cube datasets with the given shape, dimensions,
    chunksizes, and coordinate variables.

    Parameters:
    -----------
    cube_location : str
        The location of the cube dataset.
    dims : dict
        A dictionary of dimension names and their corresponding sizes.
    chunksizes : tuple
        A tuple of chunk sizes in each dimension.
    temporal_resolution : int
        The temporal resolution of the dataset (in seconds).
    spatial_resolution : int
        The spatial resolution of the dataset (in meters).
    output_metadata : dict
        A dictionary containing metadata for the output dataset.
    """

    def __init__(self,
                 cube_location: str,
                 dims: dict,
                 chunksizes: tuple,
                 temporal_resolution: int,
                 spatial_resolution: int,
                 output_metadata: dict):
        
        self.cube_location = cube_location
        self.dims_ = dims
        self.chunksizes = chunksizes
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.output_metadata = output_metadata
        
    @dask.delayed
    def _init_zarr_variable(self, 
                            IN:tuple) -> None:
        """
        Initializes a new zarr variable in the data cube.

        Parameters:
        -----------
        IN : tuple
            A tuple containing the following items:
            - name: str
                The name of the new zarr variable.
            - dims: list
                A list of dimension names for the new variable.
            - attrs: dict
                A dictionary of attributes for the new variable.
            - dtype: np.dtype
                The data type of the new variable.
        """
        name, dims, attrs, dtype = IN
        dims = [dim for dim in self.dims_ if dim in dims]
        compressor = Blosc(cname='lz4', clevel=4, shuffle=Blosc.BITSHUFFLE)
        encoding = {name: {'compressor': compressor} for var in name}
        if name not in self.cube.variables:
            xr.DataArray(
                 dask.array.full(shape  = [self.cube.coords[dim].size for dim in dims],
                                 chunks = [self.chunksizes[dim] for dim in dims], fill_value=np.nan),
                 coords = {dim:self.cube.coords[dim] for dim in dims},
                 dims   = dims,
                 name   = name,
                 attrs  = attrs
            ).chunk({dim:self.chunksizes[dim] for dim in dims}).to_dataset().to_zarr(self.cube_location, mode='a', synchronizer = self.out_cube_sync,
                                                                                     consolidated=True, encoding=encoding)

    def init_variable(self, 
                      cube_variables:dict, 
                      njobs:int = None) ->None:
        """
        Initializes all dataset variables in the SiteCube.

        Parameters:
        -----------
        cube_variables : dict
            The dataset containing the variables to be initialized. Can be either an 
            xr.Dataset or xr.DataArray object, or a dictionary with the form 
            {var_name: dims} where var_name is a string and dims is a list of 
            dimension names as strings.
        njobs : int, optional
            The number of cores to use in parallel when writing data. If not provided,
            the default value of `njobs` specified during initialization will be used.

        Raises:
        -------
        ValueError
            If the dictionary provided as `cube_variables` is not in a valid format.
        RuntimeError
            If `cube_variables` is not an xr.Dataset, xr.DataArray, dictionary, or tuple.
        """
        
        vars_to_proc = []
        for _var in cube_variables.keys():
            vars_to_proc.append( (_var, cube_variables[_var]["dims"], cube_variables[_var]["attrs"], cube_variables[_var]['dtype']) )
        futures = [self._init_zarr_variable(da_var) for da_var in vars_to_proc]
        dask.compute(*futures, num_workers= njobs)
    
        self.cube = xr.open_zarr(self.cube_location, synchronizer=self.out_cube_sync) 
    
    def initialize_data_cube(self) -> xr.Dataset:
        
        """
        Create a new empty cube with predefined coordinate variables and metadata.
 
        This function creates a new empty cube with coordinate variables and metadata 
        defined by the object's `dims_` and `output_metadata` attributes. The spatial 
        resolution is taken from the object's `spatial_resolution` attribute, and the 
        temporal resolution is taken from the object's `temporal_resolution` attribute.
 
        Returns:
        --------
        xr.Dataset
            The new empty cube as an xarray Dataset object.
 
        Raises:
        -------
        ValueError
            If the dimensions of the new cube are not valid.
        """
        coords = {dim: pd.to_datetime(self.dims_[dim]) if dim == 'time' else
                       self.dims_[dim] if dim in 'latitude' else
                       self.dims_[dim] if dim in 'longitude' else
                       self.dims_[dim] if dim in 'age_class' else                       
                       np.arange(self.dims_[dim]) if dim == 'cluster' else
                       np.arange(self.dims_[dim]) for dim in self.dims_.keys()}
        
        ds_ = xr.Dataset(data_vars={}, coords=coords, attrs= self.output_metadata)
            
        ds_.to_zarr(self.cube_location, consolidated=True, 
                    synchronizer = self.out_cube_sync)
        
    @dask.delayed
    def write_chunck(self, 
                     da: Union[xr.DataArray, xr.Dataset]) -> None:
        """
        Update a single chunk in the zarr cube. Data must be pre-sorted.

        This function is called by the `_update` function to update a specific 
        DataArray in the zarr cube. The data must be pre-sorted and aligned with the 
        dimensions of the target DataArray in the zarr cube.

        Parameters:
        -----------
        da : xr.DataArray or xr.Dataset
            The DataArray or Dataset to be updated in the zarr cube.

        Raises:
        -------
        FileExistsError
            If the `self.cube_location` already exists but is not a zarr group.
        ValueError
            If the dimensions of `da` do not match the expected dimensions of the 
            target DataArray in the zarr cube.
        RuntimeError
            If the update operation fails.
        """

        try:
            _zarr_group = zarr.open_group(self.cube_location, synchronizer = self.out_cube_sync)[da.name]

        except (IOError, ValueError, TypeError) as e:
            raise FileExistsError(
                                    f"cube_location already exists but is not a zarr group. Delete existing directory or choose a different cube_location: {self.cube_location}"
                                 ) from e
            
        # Finding the alignment indices
        alignment_indices = tuple([np.where(np.isin(self.cube[dim].values, da[dim].values))[0] for dim in da.dims])
        
        # Checking consistency in dimensions
        if len(_zarr_group.shape) != len(da.shape):
            raise ValueError(
                f"Inconsistent dimensions. Array `{da.name}` to be saved has dimensions of {da.dims}, but target dataset expected {self.cube[da.name].dims if da.name in self.cube else 'unknown'}."
                            )
        # Attempt to write the data to the Zarr group
        try:
            _zarr_group.set_orthogonal_selection(alignment_indices, da.data)
        except Exception as e:  # Consider replacing with a more specific exception if known
            raise RuntimeError(f"Failed to write variable to cube: {str(da)}") from e
        
        # Consolidate metadata for the Zarr dataset
        #zarr.consolidate_metadata(self.cube_location)

    def _update(self, 
                da:Union[xr.DataArray, xr.Dataset]) -> None:
        """
        Update the data cube with the provided data.

        This function updates the data cube with the data provided in the xarray Dataset
        or DataArray object. The update operation can be performed in parallel using
        multiple cores.

        Parameters:
        -----------
        da : Union[xr.DataArray, xr.Dataset]
            The xarray Dataset or DataArray object containing the data to be written to 
            the data cube.

        Returns:
        --------
        None

        Raises:
        -------
        RuntimeError
            If the input is not an xr.Dataset or xr.DataArray object.
        """
        if isinstance(da, xr.Dataset):
            vars_to_proc = [da[_var] for _var in (set(da.variables) - set(da.coords))]
            futures = [self.write_chunck(da_var) for da_var in vars_to_proc]
            dask.compute(*futures, num_workers=len(vars_to_proc))
            
        elif isinstance(da, xr.DataArray):
            self.write_chunck(da).compute()
        else:
            raise RuntimeError("Input must be xr.Dataset or xr.DataArray objects")
        
        