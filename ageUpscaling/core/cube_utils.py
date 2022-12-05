#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:36:53 2022

@author: simon
"""
from typing import Tuple, Sequence, Callable, Union
import inspect
import os

import numpy as np
import xarray as xr

from xcube.core.schema import CubeSchema
from xcube.core.chunkstore import ChunkStore

CubeFuncOutput = Union[xr.DataArray, np.ndarray, Sequence[Union[xr.DataArray, np.ndarray]]]
CubeFunc = Callable[..., CubeFuncOutput]

_PREDEFINED_KEYWORDS = ['input_params', 'dim_coords', 'dim_ranges']

def _inspect_cube_func(cube_func: CubeFunc, input_var_names: Sequence[str] = None):
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations \
        = inspect.getfullargspec(cube_func)
    cube_func_name = '?'
    if hasattr(cube_func, '__name__'):
        cube_func_name = cube_func.__name__
    true_args = [arg not in _PREDEFINED_KEYWORDS for arg in args]
    if False in true_args and any(true_args[true_args.index(False):]):
        raise ValueError(f'invalid cube_func {cube_func_name!r}: '
                         f'any argument must occur before any of {", ".join(_PREDEFINED_KEYWORDS)}, '
                         f'but got {", ".join(args)}')
    if not all(true_args) and varargs:
        raise ValueError(f'invalid cube_func {cube_func_name!r}: '
                         f'any argument must occur before any of {", ".join(_PREDEFINED_KEYWORDS)}, '
                         f'but got {", ".join(args)} before *{varargs}')
    num_input_vars = len(input_var_names) if input_var_names else 0
    num_args = sum(true_args)
    if varargs is None and num_input_vars != num_args:
        raise ValueError(f'invalid cube_func {cube_func_name!r}: '
                         f'expected {num_input_vars} arguments, '
                         f'but got {", ".join(args)}')
    has_input_params = 'input_params' in args or 'input_params' in kwonlyargs
    has_dim_coords = 'dim_coords' in args or 'dim_coords' in kwonlyargs
    has_dim_ranges = 'dim_ranges' in args or 'dim_ranges' in kwonlyargs
    return has_input_params, has_dim_coords, has_dim_ranges


def _gen_index_var(cube_schema: CubeSchema):
    dims = cube_schema.dims
    shape = cube_schema.shape
    chunks = cube_schema.chunks

    # noinspection PyUnusedLocal
    def get_chunk(cube_store: ChunkStore, name: str, index: Tuple[int, ...]) -> bytes:
        data = np.zeros(cube_store.chunks, dtype=np.uint64)
        data_view = data.ravel()
        if data_view.base is not data:
            raise ValueError('view expected')
        if data_view.size < cube_store.ndim * 2:
            raise ValueError('size too small')
        for i in range(cube_store.ndim):
            j1 = cube_store.chunks[i] * index[i]
            j2 = j1 + cube_store.chunks[i]
            data_view[2 * i] = j1
            data_view[2 * i + 1] = j2
        return data.tobytes()

    store = ChunkStore(dims, shape, chunks)
    store.add_lazy_array('__index_var__', '<u8', get_chunk=get_chunk)

    dataset = xr.open_zarr(store)
    index_var = dataset.__index_var__
    index_var = index_var.assign_coords(**cube_schema.coords)
    return index_var

def new_cube(name:str='test_cube',
             cube_location:str='/home/simon/Documents/science/GFZ/projects/forest-age-upscale/output',
             cube_config:dict= {}):
    """
    Create a new empty cube. Useful for creating cubes templates with
    predefined coordinate variables and metadata.
    :return: A cube instance
    """
        
    coords = cube_config['output_writer_params']['coords']
    _ds = []
    for dim in coords.keys():
        if dim == 'latitude':
            dims_ = np.arange(coords[dim][0], coords[dim][1], cube_config['spatial_resolution']) * -1 
        elif dim == 'longitude':    
            dims_ = np.arange(coords[dim][0], coords[dim][1], cube_config['spatial_resolution'])
        elif dim == 'time':    
            dims_ = np.arange(np.datetime64(coords[dim][0]), np.datetime64(coords[dim][1]), np.timedelta64(1, cube_config['temporal_resolution']))
        
        _ds.append( xr.Dataset(data_vars = {}, coords={dim:dims_}))

    _ds = xr.merge(_ds).chunk(cube_config['output_writer_params']['chunksizes'])

    _ds.to_zarr(os.path.join(cube_location, name), consolidated=True)
    
    