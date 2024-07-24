"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-FileCopyrightText: 2024 Basil Kraft
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File: plotting.py

List of useful function for plotting results
"""

import numpy as np
import xarray as xr
import scipy.stats as st
from scipy import ndimage

def calculate_pixel_area(ds, 
                         EARTH_RADIUS = 6378.160, 
                         resolution=1):
    """
    Calculate the area of a pixel in square kilometers given a latitude dataset.

    Args:
        ds (xr.DataArray): Latitude dataset in degrees.
        EARTH_RADIUS: Earth's radius in kilometers

    Returns:
        xr.DataArray: Area of the pixel in square kilometers.
    """
    # Calculate the width of each longitude slice in radians
    # Multiply by Earth's radius to get the width in kilometers
    delta_lon = np.deg2rad(resolution)  # Assuming a grid spacing of 1 degree
    width_of_longitude = EARTH_RADIUS * delta_lon
    
    # Calculate the height of each latitude slice in radians
    # Multiply by Earth's radius to get the height in kilometers
    delta_lat = np.deg2rad(resolution)  # Assuming a grid spacing of 1 degree
    height_of_latitude = EARTH_RADIUS * delta_lat
    
    # Now, calculate the area of each pixel in square kilometers
    # cos(latitude) factor accounts for the convergence of meridians at the poles
    # We need to convert latitude from degrees to radians first
    pixel_area = (width_of_longitude * height_of_latitude *
                  np.cos(np.deg2rad(ds.latitude))).broadcast_like(ds)
    
    return pixel_area


#%% Define functions
def AreaGridlatlon(lats,lons,res_lat,res_lon):
     ER          = 6378160 #Earth radius (m)
     cols        = lons.shape[0]
     londel      = np.abs(res_lon)
     lats1       = lats - res_lat/2.
     lats2       = lats + res_lat/2.
     areavec     = (np.pi/180)*ER**2 * np.abs(np.sin(lats1 * 
                                                     np.pi/180)-np.sin(lats2 * np.pi/180))*londel
     area_grid   = xr.DataArray(np.matmul(areavec[:,np.newaxis],np.ones([1, cols])), 
                               dims=['latitude', 'longitude'],
                               coords={'latitude': lats,
                                       'longitude': lons})
     return(area_grid)
 
def area_weighted_sum(data, res, scalar_area = 1, scalar_mass=1e-15):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res)
    dat_area_weighted = np.nansum(data * area_grid * scalar_area * scalar_mass)
    return dat_area_weighted

def area_weighted_mean(data, res):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res).values
    dat_area_weighted = np.nansum((data * area_grid) / np.nansum(area_grid[~np.isnan(data)]))
    return dat_area_weighted 

def filter_nan_gaussian_conserving(ds_: xr.DataArray, 
                                   length_km: float = 1000, 
                                   length_degree_longitude_equator_km: float = 112.32) -> xr.DataArray:
    """
    Apply a Gaussian filter to an xarray DataArray, preserving the total intensity 
    while considering NaN values.

    This function applies a Gaussian filter to the input DataArray `ds_`. The 
    filtering conserves the total 'intensity' (sum of the values) by redistributing
    intensity only among non-NaN pixels. The NaN values in the input DataArray remain
    NaN in the output. The Gaussian distribution weights used for intensity 
    redistribution consider only available (non-NaN) pixels. 

    The smoothing scale of the Gaussian filter is determined by `length_km`, which 
    is the physical length scale in kilometers. The sigma of the Gaussian filter is 
    calculated based on the length in degrees of longitude at the equator, given by
    `length_degree_longitude_equator_km`.

    Parameters:
    ds_ (xr.DataArray): The input DataArray to be filtered. It should contain NaN 
                        values to indicate missing data.
    length_km (float, optional): The physical length scale in kilometers for the 
                                 Gaussian filter. Default is 1000 km.
    length_degree_longitude_equator_km (float, optional): The length in degrees of 
                                                          longitude at the equator, 
                                                          used for calculating sigma 
                                                          of the Gaussian filter. 
                                                          Default is 112.32 km.

    Returns:
    xr.DataArray: A new DataArray that has been smoothed with a Gaussian filter. 
                  NaN values from the original DataArray are preserved.
    """

    sigma = length_km / length_degree_longitude_equator_km 
    
    arr = ds_.values
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr
    
    out_ = xr.DataArray(gauss, coords=ds_.coords)

    return out_

def violins(data,pos=0,bw_method=None,resolution=50,spread=1,max_num_points=100):
    """violins(data,pos=0,bw_method=None,resolution=50,spread=1)
    Jitter violin plot creater
    Takes points from a distribution and creates data for both a jitter violin and a standard violin plot.
    Parameters
    ----------
    data : numpy array
        The data to build the violin plots from
    pos : float or int
        The position the resulting violin will be centered on
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth. This can be ‘scott’, ‘silverman’, a scalar constant or a callable. If a scalar, this will be used directly as kde.factor. If a callable, it should take a gaussian_kde instance as only parameter and return a scalar. If None (default), ‘scott’ is used. See Notes for more details.
    resolution : int
        The resolution of the resulting violin plot
    spread : int or float
        The spread of the violin plots
     Returns
    -------
    pointx,pointy : numpy arrays
        The resulting data for the jitter violin plot (use with pl.scatter)
    fillx,filly : numpy array
        The resulting data for a standard violin plot (use with pl.fill_between)
    """
    if data.size>max_num_points:
        data = np.random.choice(data,size=max_num_points,replace=False)
    kde    = st.gaussian_kde(data,bw_method=bw_method)
    pointx = data
    pointy = kde.pdf(pointx)
    pointy = pointy/(2*pointy.max())
    fillx  = np.linspace(data.min(),data.max(),resolution)
    filly  = kde.pdf(fillx)
    filly  = filly/(2*filly.max())
    pointy = pos+np.where(np.random.rand(pointx.shape[0])>0.5,-1,1)*np.random.rand(pointx.shape[0])*pointy*spread
    filly  = (pos-filly*spread,pos+filly*spread)
    return(pointx,pointy,fillx,filly)

def blur_func(num, sigma):
    """Kernel density function
    Currently Gaussian, but this can change to need.
    """
    assert num % 2 == 1, 'Only with odd numbers!'
    kernel = np.zeros(num)
    mu = num // 2
    for x in range(num):
        # Could vectorize
        density = (1/(sigma * (2 * np.pi) ** 0.5) * np.exp(-1/2 * ((x - mu) ** 2) / sigma ** 2))
        kernel[x] = density
    return kernel

def get_kernel(sigma_x, sigma_y, kernel_size):
    """Function to create a 2d kernel based on the creation of 2 1d kernels. """
    kernel_x = blur_func(kernel_size, sigma_x)
    kernel_y = blur_func(kernel_size, sigma_y)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum() # Normalize to 1
    return kernel

def apply_blur(NEE_2010, kernel_size=5, sigmax=2):
    HALF_KERNEL = int(np.ceil(kernel_size / 2))
    
    # Calculate the number of pixels per latitude bin
    global_lats = NEE_2010.latitude.values
    sigma_per_lat = 1/np.cos(global_lats / 90) * sigmax

    newmap = NEE_2010.values.copy()
    for i in range(NEE_2010.shape[0]):
        for j in range(NEE_2010.shape[1]):
            
            sigmay = sigma_per_lat[i]
            kernel = get_kernel(sigmax, sigmay, kernel_size)
            starti = max(0, i - (HALF_KERNEL - 1))
            endi = min(newmap.shape[0], i + HALF_KERNEL)
            
            startj = max(0, j - (HALF_KERNEL - 1))
            endj = min(newmap.shape[1], j + HALF_KERNEL)

            if i - HALF_KERNEL < 0:
                kernel = kernel[-(i - HALF_KERNEL + 1):, :]
            elif i + HALF_KERNEL > newmap.shape[0]:
                kernel = kernel[:newmap.shape[0] - (i + HALF_KERNEL), :]
                
            if j - HALF_KERNEL < 0:
                kernel = kernel[:, -(j - HALF_KERNEL + 1):]
            elif j + HALF_KERNEL > newmap.shape[1]:
                kernel = kernel[:, :newmap.shape[1] - (j + HALF_KERNEL)]

            a = newmap[starti: endi,
                    startj: endj]

            assert a.shape == kernel.shape

            newmap[i, j] = np.average(a, weights=kernel)
    
    return xr.DataArray(newmap, coords=NEE_2010.coords)

def rolling_window(data, window_size=10):
    latitudes = data.latitude.values
    longitudes = data.longitude.values

    mean_values = []

    for lat in range(int(latitudes.min()), int(latitudes.max()) - window_size + 1):
        for lon in range(int(longitudes.min()), int(longitudes.max()) - window_size + 1):
            # Select data in the current window
            window_data = data.sel(latitude=slice(lat + window_size, lat),
                                   longitude=slice(lon, lon + window_size))
            # Calculate area-weighted mean for this window
            window_mean = window_data.mean(dim=('latitude', 'longitude')).values
            mean_values.append([window_mean])
    
    return np.concatenate(mean_values)