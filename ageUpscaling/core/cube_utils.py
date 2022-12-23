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
@Desc    :   A method class for creating new cube datasets.
"""
import atexit
import os
from abc import ABC
from typing import Optional, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import shutil

#from ageUpscaling.utils.utilities import async_run

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class ComputeCube(ABC):
    """A schema for creating new cube datasets.
    
    This class can be used to create new cube datasets with the given shape, dimensions,
    chunksizes, and coordinate variables.
    
    Parameters:
    -----------
    cube_location: str
        The location of the cube dataset.
    dims: dict
        A dictionary of dimension names and their corresponding sizes.
    chunksizes: tuple
        A tuple of chunk sizes in each dimension.
    temporal_resolution: int
        The temporal resolution of the dataset (in seconds).
    spatial_resolution: int
        The spatial resolution of the dataset (in meters).
    output_metadata: dict
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
        
    def _init_zarr_variable(self, 
                            IN:tuple) -> None:
        """Initializes a new zarr variable in the data cube.
        
        Parameters:
        -----------
        IN: tuple
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
        if name not in self.cube.variables:
            xr.DataArray(
                 dask.array.full(shape  = [self.cube.coords[dim].size for dim in dims],
                                 chunks = [self.chunksizes[dim] for dim in dims], fill_value=np.nan),
                 coords = {dim:self.cube.coords[dim] for dim in dims},
                 dims   = dims,
                 name   = name,
                 attrs  = attrs
            ).chunk({dim:self.chunksizes[dim] for dim in dims}).to_dataset().to_zarr(self.cube_location, mode='a')

        
    def init_variable(self, 
                      dataset: Union[xr.DataArray, xr.Dataset], 
                      njobs:int = None, 
                      parallel:bool = False) ->None:
        """Initializes all dataset variables in the SiteCube.
        
        Parameters:
        -----------
        dataset: xr.Dataset, xr.DataArray, dict, or tuple
            The dataset containing the variables to be initialized. Can be either an 
            xr.Dataset or xr.DataArray object, or a dictionary with the form 
            {var_name: dims} where var_name is a string and dims is a list of 
            dimension names as stings.
        njobs: int, optional
            The number of cores to use in parallel when writing data. If not provided,
            the default value of `njobs` specified during initialization will be used.
        parallel: bool, optional
            A flag indicating whether to run the initialization in parallel. If set to
            `True`, the function will use the specified number of cores to write data 
            in parallel. If set to `False`, the function will write data sequentially.
            
        Raises:
        -------
        ValueError:
            If the dictionary provided as `dataset` is not in a valid format.
        RuntimeError:
            If `dataset` is not an xr.Dataset, xr.DataArray, dictionary, or tuple.
        """
        if dataset.__class__ is xr.DataArray:
            self._init_zarr_variable( (dataset.name, dataset.dims, dataset.attrs, dataset.dtype) )
        elif dataset.__class__ is xr.Dataset:
            to_proc = []
            for _var in set(dataset.variables) - set(dataset.coords):
                to_proc.append( (_var, dataset[_var].dims, dataset[_var].attrs, dataset[_var].dtype) )
            futures = [dask.delayed(self._init_zarr_variable)(da_var) for da_var in to_proc]
            dask.compute(*futures, num_workers= njobs)
        else:
            raise RuntimeError("dataset must be xr.Dataset, xr.DataArray")
            # if parallel:
            #     out = async_run(self._init_zarr_variable, to_proc, njobs) # issue with cascading multiprocessing, maybe fix in the future.
            # else:
            #     out = list(map(self._init_zarr_variable, to_proc))
        
        self.cube = xr.open_zarr(self.cube_location) 
    
    def new_cube(self) -> xr.Dataset:
        """Create a new empty cube with predefined coordinate variables and metadata.
        
        This function creates a new empty cube with coordinate variables and metadata 
        defined by the object's `dims_` and `output_metadata` attributes. The spatial 
        resolution is taken from the object's `spatial_resolution` attribute, and the 
        temporal resolution is taken from the object's `temporal_resolution` attribute.
        
        Returns:
        --------
        xr.Dataset:
            The new empty cube as an xarray Dataset object.
        
        Raises:
        -------
        ValueError:
            If the dimensions of the new cube are not valid.
        """ 
        coords = {dim: np.arange(self.dims_[dim][0], self.dims_[dim][1], self.spatial_resolution) * -1 if dim in 'latitude' else
                       np.arange(self.dims_[dim][0], self.dims_[dim][1], self.spatial_resolution) if dim in 'longitude' else
                       np.array(pd.to_datetime(self.dims_[dim])) if dim == 'time' else
                       np.arange(self.dims_[dim]) + 1 if dim == 'cluster' else
                       np.arange(self.dims_[dim]) for dim in self.dims_.keys()}
        
        ds_ = xr.Dataset(data_vars={}, coords=coords, attrs= self.output_metadata)
            
        ds_.to_zarr(self.cube_location, consolidated=True)
        
    def _update_cube_DataArray(self, 
                               da: Union[xr.DataArray, xr.Dataset],
                               sync: Optional[zarr.ProcessSynchronizer] = None) -> None:
        """Update a single DataArray in the zarr cube. Data must be pre-sorted.
        
        This function is called by the `update_cube` function to update a specific 
        DataArray in the zarr cube. The data must be pre-sorted and aligned with the 
        dimensions of the target DataArray in the zarr cube.
        
        Parameters:
        -----------
        da: xr.DataArray or xr.Dataset
            The DataArray or Dataset to be updated in the zarr cube.
        sync: zarr.Synchronizer, optional
            An optional synchronizer to use when updating the data. If not provided,
            the default synchronizer specified during initialization will be used.
            
        Raises:
        -------
        FileExistsError:
            If the `self.cube_location` already exists but is not a zarr group.
        ValueError:
            If the dimensions of `da` do not match the expected dimensions of the 
            target DataArray in the zarr cube.
        RuntimeError:
            If the update operation fails.
        """
        
        if sync is None:
            sync = zarr.ProcessSynchronizer('.zarrsync')
        
        try:
            _zarr = zarr.open_group(self.cube_location, synchronizer = sync)[da.name]
        except (IOError, ValueError, TypeError) as e:
            raise FileExistsError("cube_location already exists but is not a zarr group. Delete existing directory or choose a different cube_location: "+self.cube_location) from e
        
        idxs = tuple([np.where( np.isin(self.cube[dim].values, da[dim].values ) )[0] for dim in da.dims])
        print(idxs)
        
        if len(_zarr.shape) != len(da.shape):
            raise ValueError("Inconsistent dimensions. Array `{0}` to be saved has dimensions of {1}, but target dataset expected {2}.".format(da.name, da.dims, self.cube[da.name].dims))
        try:
            _zarr.set_orthogonal_selection(idxs, da.data)
        except Exception as e:
            raise RuntimeError("Failed to write variable to cube: " + str(da)) from e
        
    def _update(self, 
                da:Union[xr.DataArray, xr.Dataset], 
                njobs:int=None) -> None:
        """Update the data cube with the provided data.
    
        This function updates the data cube with the data provided in the xarray Dataset
        or DataArray object. The update operation can be performed in parallel using
        multiple cores.
    
        Parameters:
        -----------
        da: Union[xr.DataArray, xr.Dataset]
            The xarray Dataset or DataArray object containing the data to be written to 
            the data cube.
        njobs: int, optional
            The number of cores to use when performing the update operation in parallel.
            The default value is None, which will use the default number of cores specified
            during the initialization of the SiteCube object.
    
        Returns:
        --------
        None
    
        Raises:
        -------
        RuntimeError:
            If the input is not an xr.Dataset or xr.DataArray object.
        """
        if da.__class__ is xr.Dataset:
            to_proc = [da[_var] for _var in (set(da.variables) - set(da.coords))]
            futures = [dask.delayed(self._update_cube_DataArray)(da_var) for da_var in to_proc]
            dask.compute(*futures, num_workers= njobs)
            #_ = async_run(self._update_cube_DataArray, to_proc, njobs)
        elif da.__class__ is xr.DataArray:
            self._update_cube_DataArray(da)
        else:
            raise RuntimeError("Input must be xr.Dataset or xr.DataArray objects")