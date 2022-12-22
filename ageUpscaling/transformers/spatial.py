#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   spatial.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   Methods for spatial operation
"""
from typing import Union

import xarray as xr

def interpolate_worlClim(source_ds:Union[xr.DataArray, xr.Dataset], 
                         target_ds:Union[xr.DataArray, xr.Dataset],
                         method:str = 'linear') -> Union[xr.DataArray, xr.Dataset]:    
    """
    Interpolates the source dataset to the coordinates of the target dataset.
    
    Parameters
    ----------
    source_ds: xr.DataArray or xr.Dataset
        The source dataset to be interpolated.
    target_ds: xr.DataArray or xr.Dataset
        The target dataset with the coordinates to interpolate to.
    method: str, optional
        The interpolation method to use. Supported methods are 'linear' (default), 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.
        
    Returns
    -------
    resampled: xr.DataArray or xr.Dataset
        The interpolated source dataset.
    
    Raises
    ------
    ValueError
        If the interpolation fails in either the latitude or longitude axis.
    """
    
    resampled = source_ds.interp(latitude = target_ds.latitude, 
                                 longitude = target_ds.longitude,
                                 method=method)
    
    if not (resampled.latitude.data == target_ds.latitude.data).all():
        raise ValueError("Failed to interpolate in the latitude axis")
        
    if not (resampled.longitude.data == target_ds.longitude.data).all():
        raise ValueError("Failed to interpolate in the longitude axis")
    
    return resampled
