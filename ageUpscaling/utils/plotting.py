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

def calculate_pixel_area(ds, 
                         EARTH_RADIUS = 6371.0, 
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