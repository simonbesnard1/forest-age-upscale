#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:36:53 2022

@author: simon
"""
from typing import Tuple, Sequence, Callable, Union
import inspect

import numpy as np
import xarray as xr
import pandas as pd 
import itertools
import pyproj

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

def new_cube(title = 'test cube',
             cube_location='/home/simon/Documents/test',
             width=360,
             height=180,
             x_name='lon',
             y_name='lat',
             x_dtype='float64',
             y_dtype=None,
             x_units='degrees_east',
             y_units='degrees_north',
             x_res=1.0,
             y_res=None,
             x_start=-180.0,
             y_start=-90.0,
             inverse_y=False,
             time_name='time',
             time_dtype='datetime64[s]',
             time_units='seconds since 1970-01-01T00:00:00',
             time_calendar='proleptic_gregorian',
             time_periods=5,
             time_freq="D",
             time_start='2010-01-01T00:00:00',
             use_cftime=False,
             drop_bounds=False,
             variables=None,
             crs=None,
             crs_name=None):
    """
    Create a new empty cube. Useful for creating cubes templates with
    predefined coordinate variables and metadata. The function is also
    heavily used by xcube's unit tests.
    The values of the *variables* dictionary can be either constants,
    array-like objects, or functions that compute their return value from
    passed coordinate indexes. The expected signature is:::
        def my_func(time: int, y: int, x: int) -> Union[bool, int, float]
    :param title: A title. Defaults to 'Test Cube'.
    :param width: Horizontal number of grid cells. Defaults to 360.
    :param height: Vertical number of grid cells. Defaults to 180.
    :param x_name: Name of the x coordinate variable. Defaults to 'lon'.
    :param y_name: Name of the y coordinate variable. Defaults to 'lat'.
    :param x_dtype: Data type of x coordinates. Defaults to 'float64'.
    :param y_dtype: Data type of y coordinates. Defaults to 'float64'.
    :param x_units: Units of the x coordinates. Defaults to 'degrees_east'.
    :param y_units: Units of the y coordinates. Defaults to 'degrees_north'.
    :param x_start: Minimum x value. Defaults to -180.
    :param y_start: Minimum y value. Defaults to -90.
    :param x_res: Spatial resolution in x-direction. Defaults to 1.0.
    :param y_res: Spatial resolution in y-direction. Defaults to 1.0.
    :param inverse_y: Whether to create an inverse y axis. Defaults to False.
    :param time_name: Name of the time coordinate variable. Defaults to 'time'.
    :param time_periods: Number of time steps. Defaults to 5.
    :param time_freq: Duration of each time step. Defaults to `1D'.
    :param time_start: First time value. Defaults to '2010-01-01T00:00:00'.
    :param time_dtype: Numpy data type for time coordinates.
        Defaults to 'datetime64[s]'.
        If used, parameter 'use_cftime' must be False.
    :param time_units: Units for time coordinates.
        Defaults to 'seconds since 1970-01-01T00:00:00'.
    :param time_calendar: Calender for time coordinates.
        Defaults to 'proleptic_gregorian'.
    :param use_cftime: If True, the time will be given as data types
        according to the 'cftime' package. If used, the time_calendar
        parameter must be also be given with an appropriate value
        such as 'gregorian' or 'julian'. If used, parameter 'time_dtype'
        must be None.
    :param drop_bounds: If True, coordinate bounds variables are not created.
        Defaults to False.
    :param variables: Dictionary of data variables to be added.
        None by default.
    :param crs: pyproj-compatible CRS string or instance
        of pyproj.CRS or None
    :param crs_name: Name of the variable that will
        hold the CRS information. Ignored, if *crs* is not given.
    :return: A cube instance
    """
    y_dtype = y_dtype if y_dtype is not None else y_dtype
    y_res = y_res if y_res is not None else x_res
    if width < 0 or height < 0 or x_res <= 0.0 or y_res <= 0.0:
        raise ValueError()
    if time_periods < 0:
        raise ValueError()

    if use_cftime and time_dtype is not None:
        raise ValueError('If "use_cftime" is True,'
                         ' "time_dtype" must not be set.')

    x_is_lon = x_name == 'lon' or x_units == 'degrees_east'
    y_is_lat = y_name == 'lat' or y_units == 'degrees_north'

    x_end = x_start + width * x_res
    y_end = y_start + height * y_res

    x_res_05 = 0.5 * x_res
    y_res_05 = 0.5 * y_res

    x_data = np.linspace(x_start + x_res_05, x_end - x_res_05,
                         width, dtype=x_dtype)
    y_data = np.linspace(y_start + y_res_05, y_end - y_res_05,
                         height, dtype=y_dtype)

    x_var = xr.DataArray(x_data, dims=x_name, attrs=dict(units=x_units))
    y_var = xr.DataArray(y_data, dims=y_name, attrs=dict(units=y_units))
    if inverse_y:
        y_var = y_var[::-1]

    if x_is_lon:
        x_var.attrs.update(long_name='longitude',
                           standard_name='longitude')
    else:
        x_var.attrs.update(long_name='x coordinate of projection',
                           standard_name='projection_x_coordinate')
    if y_is_lat:
        y_var.attrs.update(long_name='latitude',
                           standard_name='latitude')
    else:
        y_var.attrs.update(long_name='y coordinate of projection',
                           standard_name='projection_y_coordinate')

    if use_cftime:
        time_data_p1 = xr.cftime_range(start=time_start,
                                       periods=time_periods + 1,
                                       freq=time_freq,
                                       calendar=time_calendar).values
    else:
        time_data_p1 = pd.date_range(start=time_start,
                                     periods=time_periods + 1,
                                     freq=time_freq).values
        time_data_p1 = time_data_p1.astype(dtype=time_dtype)

    time_delta = time_data_p1[1] - time_data_p1[0]
    time_data = time_data_p1[0:-1] + time_delta // 2
    time_var = xr.DataArray(time_data, dims=time_name)
    time_var.encoding['units'] = time_units
    time_var.encoding['calendar'] = time_calendar

    coords = {x_name: x_var, y_name: y_var, time_name: time_var}

    attrs = dict(Conventions="CF-1.7",
                 title=title,
                 time_coverage_start=str(time_data_p1[0]),
                 time_coverage_end=str(time_data_p1[-1]))

    if x_is_lon:
        attrs.update(dict(geospatial_lon_min=x_start,
                          geospatial_lon_max=x_end,
                          geospatial_lon_units=x_units))

    if y_is_lat:
        attrs.update(dict(geospatial_lat_min=y_start,
                          geospatial_lat_max=y_end,
                          geospatial_lat_units=y_units))

    data_vars = {}
    if variables:
        dims = (time_name, y_name, x_name)
        shape = (time_periods, height, width)
        size = time_periods * height * width
        for var_name, data in variables.items():
            if isinstance(data, xr.DataArray):
                data_vars[var_name] = data
            elif isinstance(data, int) \
                    or isinstance(data, float) \
                    or isinstance(data, bool):
                data_vars[var_name] = xr.DataArray(
                    np.full(shape, data), dims=dims
                )
            elif callable(data):
                func = data
                data = np.zeros(shape)
                for index in itertools.product(*map(range, shape)):
                    data[index] = func(*index)
                data_vars[var_name] = xr.DataArray(data, dims=dims)
            elif data is None:
                data_vars[var_name] = xr.DataArray(
                    np.random.uniform(0.0, 1.0, size).reshape(shape),
                    dims=dims
                )
            else:
                data_vars[var_name] = xr.DataArray(data, dims=dims)

    if isinstance(crs, str):
        crs = pyproj.CRS.from_string(crs)

    if isinstance(crs, pyproj.CRS):
        crs_name = crs_name or 'crs'
        for v in data_vars.values():
            v.attrs['grid_mapping'] = crs_name
        data_vars[crs_name] = xr.DataArray(0, attrs=crs.to_cf())

    _ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    
    _ds.to_zarr(cube_location, consolidated=True)
