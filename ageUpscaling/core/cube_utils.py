#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:36:53 2022

@author: simon
"""
from typing import Union
from abc import ABC
import os
import atexit

from ageUpscaling.utils.utilities import async_run

import numpy as np
import xarray as xr
import dask
import pandas as pd

import zarr
import shutil

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class ComputeCube(ABC):
    """
    A schema that can be used to create new cube datasets.
    The given *shape*, *dims*, and *chunks*, *coords* apply to all data variables.

    :param shape: A tuple of dimension sizes.
    :param coords: A dictionary of coordinate variables. Must have values for all *dims*.
    :param dims: A sequence of dimension names. Defaults to ``('time', 'lat', 'lon')``.
    :param chunks: A tuple of chunk sizes in each dimension.
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
        
    def _init_zarr_variable(self, IN):
        """
        Initializes a new zarr variable in the data cube.
        """
        name, dims, attrs, dtype = IN
        #_cube = xr.open_zarr(self.cube_location, synchronizer=synchronizer)
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

            
    def init_variable(self, dataset, njobs = None, parallel = False):
        """init_variable(dataset)
        
        Initializes all dataset variables in the SiteCube.
        
        Parameters
        ----------
        dataset : xr.Dataset, xr.DataArray, dictionary, or tuple
            Must be either xr.Dataset or xr.DataArray objects,
            or a dictionary with of the form {var_name: dims}
            where var_name is a string and dims is a list of
            dimension names as stings.
            
        njobs : int, default is None
            Number of cores to use in parallel when writing data, None will
            result in the default njobs as defined in during initialization.
        
        """
        if njobs is None:
            njobs = self.njobs
        #self._init_cube()
        if type(dataset) is xr.DataArray:
            self._init_zarr_variable( (dataset.name, dataset.dims, dataset.attrs, dataset.dtype) )
        elif type(dataset) is tuple:
            self._init_zarr_variable( dataset )
        else:
            to_proc = []
            if type(dataset) is xr.Dataset:
                for _var in set(dataset.variables) - set(dataset.coords):
                    to_proc.append( (_var, dataset[_var].dims, dataset[_var].attrs, dataset[_var].dtype) )
            elif type(dataset) is dict:
                for k,v in dataset.items():
                    if type(v) is str:
                        to_proc.append( (k, v, None, np.float64) )
                    elif type(v) is dict:
                        to_proc.append( (k, v['dims'], v['attrs'], np.float64) )
                    elif len(v) ==2:
                        to_proc.append( (k, v[0], v[1], np.float64) )
                    else:
                        raise ValueError("key:value pair must be constructed as one of: var_name:(dims, attrs), var_name:{dims:dims, attrs:attrs}, or var_name:dim")
            else:
                raise RuntimeError("dataset must be xr.Dataset, xr.DataArray, dictionary, or tuple")
            if parallel:
                out = async_run(self._init_zarr_variable, to_proc, njobs) # issue with cascading multiprocessing, maybe fix in the future.
            else:
                out = list(map(self._init_zarr_variable, to_proc))
        
        self.cube = xr.open_zarr(self.cube_location) 
    
    def new_cube(self) -> xr.Dataset:
        """
        Create a new empty cube with predefined coordinate variables and metadata.
    
        The coordinates and metadata are taken from the object's `dims_` and `output_metadata` attributes, respectively.
        The spatial resolution is taken from the object's `spatial_resolution` attribute.
        The temporal resolution is taken from the object's `temporal_resolution` attribute.
    
        :return: A cube instance
        """  
        coords = {dim: np.arange(self.dims_[dim][0], self.dims_[dim][1], self.spatial_resolution) * -1 if dim in 'latitude' else
                       np.arange(self.dims_[dim][0], self.dims_[dim][1], self.spatial_resolution) if dim in 'longitude' else
                       #np.arange(np.datetime64(self.dims_[dim][0]), np.datetime64(self.dims_[dim][1]), np.timedelta64(1, self.temporal_resolution)) if dim == 'time' else
                       np.array(pd.to_datetime(self.dims_[dim])) if dim == 'time' else
                       np.arange(self.dims_[dim]) + 1 if dim == 'cluster' else
                       np.arange(self.dims_[dim]) if dim == 'sample' else 
                       np.arange(self.dims_[dim]) for dim in self.dims_.keys()}
        
        ds_ = xr.Dataset(data_vars={}, coords=coords, attrs= self.output_metadata)
            
        ds_.to_zarr(self.cube_location, consolidated=True)
        
    def _update_cube_DataArray(self, 
                               da: Union[xr.DataArray, xr.Dataset],
                               sync = None):
        """
        Updates a single DataArray in the zarr cube. Data must be pre-sorted.
        Inputs to the `update_cube` function ultimately are passed here.
        """
        
        if sync is None:
            sync = zarr.ProcessSynchronizer('.zarrsync')
        
        try:
            _zarr = zarr.open_group(self.cube_location, synchronizer = sync)[da.name]
        except (IOError, ValueError, TypeError) as e:
            raise FileExistsError("cube_location already exists but is not a zarr group. Delete existing directory or choose a different cube_location: "+self.cube_location) from e
        
        idxs = tuple([np.where( np.isin(self.cube[dim].values, da[dim].values ) )[0] for dim in da.dims])
        
        if len(_zarr.shape) != len(da.shape):
            raise ValueError("Inconsistent dimensions. Array `{0}` to be saved has dimensions of {1}, but target dataset expected {2}.".format(da.name, da.dims, self.cube[da.name].dims))
        try:
            _zarr.set_orthogonal_selection(idxs, da.data)
        except Exception as e:
            raise RuntimeError("Failed to write variable to cube: "+str(da)) from e
        
    def _update(self, 
                da:Union[xr.DataArray, xr.Dataset], 
                njobs:int=None):
        """
        Handles both Datasets and DataArrays.
        """
        if njobs is None:
            njobs = self.njobs
        update_function = self._update_cube_DataArray

        if isinstance(da, xr.Dataset):
            to_proc = [da[_var] for _var in (set(da.variables) - set(da.coords))]
            _ = async_run(update_function, to_proc, njobs)
        elif isinstance(da, xr.DataArray):
            self._update_cube_DataArray(da)
        else:
            raise RuntimeError("Input must be xr.Dataset or xr.DataArray objects")