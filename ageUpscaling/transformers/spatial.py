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

import numpy as np
import xarray as xr


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Infer cell edges from a monotonic array of cell-center coordinates.

    Works for either ascending or descending `centers` (e.g. latitude is
    typically descending in these products); the returned edges preserve
    that same direction, so callers can pass them straight to `.sel(slice(
    edges[0], edges[-1]))` without worrying about sort order.
    """
    centers = np.asarray(centers, dtype=float)
    mid = (centers[:-1] + centers[1:]) / 2.0
    first_edge = centers[0] - (mid[0] - centers[0])
    last_edge = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first_edge], mid, [last_edge]])


def block_mean_regrid(
    source_da: xr.DataArray,
    target_ds: Union[xr.DataArray, xr.Dataset],
) -> xr.DataArray:
    """Area/block-mean aggregate a finer-resolution grid onto a coarser one.

    Unlike `interpolate_worlClim` (point interpolation, meant for
    coarse-to-fine downscaling such as 1km climate normals onto a 100m
    grid), this is for the opposite direction: aggregating a *finer*
    source grid onto a *coarser* target grid, where every source pixel
    should contribute to the target cell it physically falls inside (e.g.
    30m canopy height onto a ~100m prediction grid), rather than
    subsampling or linearly blending a handful of source points per target
    pixel.

    Every source pixel is assigned to exactly one target cell by comparing
    its center coordinate against the target grid's cell edges (inferred
    from the target center spacing), then averaged per cell via a
    bincount-based reduction (vectorised, no per-cell Python loop).

    Parameters
    ----------
    source_da : xr.DataArray
        Finer-resolution source data with 'latitude'/'longitude' dims.
        Loaded eagerly (`.compute()`) internally for the selected window,
        so callers should pre-select a reasonably sized spatial window
        before calling this (e.g. one prediction chunk plus a small
        buffer), not the full global array.
    target_ds : xr.DataArray or xr.Dataset
        Target grid whose 'latitude'/'longitude' coordinates define the
        output cell centers.

    Returns
    -------
    xr.DataArray
        `source_da` block-averaged onto `target_ds`'s latitude/longitude
        grid. Target cells with no finite source pixels are NaN.
    """
    target_lat = target_ds.latitude.values
    target_lon = target_ds.longitude.values

    lat_edges = _edges_from_centers(target_lat)
    lon_edges = _edges_from_centers(target_lon)

    source = source_da.sel(
        latitude=slice(lat_edges[0], lat_edges[-1]),
        longitude=slice(lon_edges[0], lon_edges[-1]),
    ).compute()

    src_lat = source.latitude.values
    src_lon = source.longitude.values
    src_vals = source.values

    # np.digitize requires monotonically increasing bins; normalise to
    # ascending internally and map the resulting bin indices back.
    lat_ascending = lat_edges[0] < lat_edges[-1]
    lon_ascending = lon_edges[0] < lon_edges[-1]
    lat_edges_asc = lat_edges if lat_ascending else lat_edges[::-1]
    lon_edges_asc = lon_edges if lon_ascending else lon_edges[::-1]

    n_lat_bins = len(target_lat)
    n_lon_bins = len(target_lon)

    lat_bin = np.digitize(src_lat, lat_edges_asc) - 1
    lon_bin = np.digitize(src_lon, lon_edges_asc) - 1
    if not lat_ascending:
        lat_bin = n_lat_bins - 1 - lat_bin
    if not lon_ascending:
        lon_bin = n_lon_bins - 1 - lon_bin

    lat_bin_2d, lon_bin_2d = np.meshgrid(lat_bin, lon_bin, indexing="ij")
    valid = (
        (lat_bin_2d >= 0) & (lat_bin_2d < n_lat_bins) &
        (lon_bin_2d >= 0) & (lon_bin_2d < n_lon_bins) &
        np.isfinite(src_vals)
    )

    flat_idx = lat_bin_2d[valid] * n_lon_bins + lon_bin_2d[valid]
    vals = src_vals[valid]

    n_cells = n_lat_bins * n_lon_bins
    sums = np.bincount(flat_idx, weights=vals, minlength=n_cells)
    counts = np.bincount(flat_idx, minlength=n_cells)
    means = np.full(n_cells, np.nan)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    return xr.DataArray(
        means.reshape(n_lat_bins, n_lon_bins),
        coords={"latitude": target_lat, "longitude": target_lon},
        dims=["latitude", "longitude"],
    )


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
