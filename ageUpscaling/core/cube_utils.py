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

import zarr
import shutil
synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class ComputeCube(ABC):
    """
    A schema that can be used to create new xcube datasets.
    The given *shape*, *dims*, and *chunks*, *coords* apply to all data variables.

    :param shape: A tuple of dimension sizes.
    :param coords: A dictionary of coordinate variables. Must have values for all *dims*.
    :param dims: A sequence of dimension names. Defaults to ``('time', 'lat', 'lon')``.
    :param chunks: A tuple of chunk sizes in each dimension.
    """

    def __init__(self,
                 cube_config:dict= None):
        
        self.dims_= cube_config['output_writer_params']['dims']
        self.variables = cube_config['output_variables']
        self.cube_location = cube_config['cube_location']
        self.chunksizes = cube_config['output_writer_params']['chunksizes']
        self.temporal_resolution = cube_config['temporal_resolution']
        self.spatial_resolution = cube_config['spatial_resolution']
        
    def init_variable(self, dataset, njobs=None, parallel=True):
        """init_variable(dataset)

        Initializes all dataset variables in the Cube.

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
        #TODO
        
        if njobs is None:
            njobs = self.njobs
        self._init_cube()
        if type(dataset) is xr.DataArray:
            self._init_zarr_variable((dataset.name, dataset.dims, dataset.attrs, dataset.dtype))
        elif type(dataset) is tuple:
            self._init_zarr_variable(dataset)
        else:
            to_proc = []
            if type(dataset) is xr.Dataset:
                for _var in set(dataset.variables) - set(dataset.coords):
                    to_proc.append((_var, dataset[_var].dims, dataset[_var].attrs, dataset[_var].dtype))
            elif type(dataset) is dict:
                for k, v in dataset.items():
                    if type(v) is str:
                        to_proc.append((k, v, None, np.float64))
                    elif type(v) is dict:
                        to_proc.append((k, v['dims'], v['attrs'], np.float64))
                    elif len(v) == 2:
                        to_proc.append((k, v[0], v[1], np.float64))
                    else:
                        raise ValueError(
                            "key:value pair must be constructed as one of: var_name:(dims, attrs),\
                                var_name:{dims:dims, attrs:attrs}, or var_name:dim")
            else:
                raise RuntimeError("dataset must be xr.Dataset, xr.DataArray, dictionary, or tuple")
            _ = async_run(self._init_zarr_variable, to_proc, njobs)
        self.cube = xr.open_zarr(self.cube_location)
    
    def new_cube(self) -> xr.Dataset:
        """
        Create a new empty cube. Useful for creating cubes templates with
        predefined coordinate variables and metadata.
        :return: A cube instance
        """
            
        ds_ = []
        coords = {}
        for dim in self.dims_.keys():
            if dim == 'latitude':
                dim_ = np.arange(self.dims_[dim][0], self.dims_[dim][1], self.spatial_resolution) * -1 
            elif dim == 'longitude':    
                dim_ = np.arange(self.dims_[dim][0], self.dims_[dim][1], self.spatial_resolution)
            elif dim == 'time':    
                dim_ = np.arange(np.datetime64(self.dims_[dim][0]), np.datetime64(self.dims_[dim][1]),
                                  np.timedelta64(1, self.temporal_resolution))
            coords.update({dim: dim_})
        
        dims = coords.keys()
        shape = [v.size for v in coords.values() if len(v.shape) > 0]
        data_vars = {}   
        
        if self.variables:
            for var_name in self.variables:
                
                data_vars[var_name] = xr.DataArray(dask.array.full(shape=shape,
                                                                    chunks=self.chunksizes, 
                                                                    fill_value=np.nan),
                                                    coords=coords,
                                                    dims=dims,
                                                    name=var_name
                                                ).chunk(self.chunksizes)
                
            ds_ = xr.Dataset(data_vars=data_vars, coords=coords)
        
        else :
            ds_ = xr.Dataset(data_vars={}, coords=coords)        
            
        ds_.to_zarr(self.cube_location, consolidated=True)
        
    def _update_cube_DataArray(self, 
                               da: Union[xr.DataArray, xr.Dataset]):
        """
        Updates a single DataArray in the zarr cube. Data must be pre-sorted.
        Inputs to the `update_cube` function ultimately are passed here.
        """
        
        try:
            _zarr = zarr.open_group(self.cube_location, synchronizer = synchronizer)[da.name]
        except ValueError as e:
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
        if type(da) is xr.Dataset:
            to_proc = [da[_var] for _var in (set(da.variables) - set(da.coords))]
            _ = async_run(update_function, to_proc, njobs)
        elif type(da) is xr.DataArray:
            update_function(da)
        else:
            raise RuntimeError("Input must be xr.Dataset or xr.DataArray objects")