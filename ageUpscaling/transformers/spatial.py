#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File    :   spatial.py
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
