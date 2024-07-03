#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:09:32 2024

@author: simon
"""
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def get_coordinates_of_class_center(data_array, class_value):
    """
    Get the geographic center coordinates of a specified class in a DataArray.

    :param data_array: xarray.DataArray with latitude and longitude dimensions.
    :param class_value: The class value to find the center of.
    :return: (latitude, longitude) of the class center.
    """
    # Mask the array to include only cells of the specified class
    mask = data_array == class_value

    # Check if there are any cells of the specified class
    if mask.sum() == 0:
        return None, None  # Return None if the class is not present

    # Find indices where the mask is True
    lat_indices, lon_indices = np.where(mask)

    # Get the coordinates of the cells
    latitudes = data_array.latitude.values
    longitudes = data_array.longitude.values

    # Calculate the mean latitude and longitude using the indices
    mean_lat = latitudes[lat_indices].mean()
    mean_lon = longitudes[lon_indices].mean()

    return mean_lat, mean_lon

#%% Define transcom regions
GFED_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/GFED_regions/GFED_regions_360_180_v1.nc').basis_regions
GFED_regions = GFED_regions.where((GFED_regions == 9) | (GFED_regions == 8))
GFED_regions = GFED_regions.where((GFED_regions ==9) | (np.isnan(GFED_regions)), 5)
GFED_regions = GFED_regions.where((GFED_regions ==5) | (np.isnan(GFED_regions)), 6)
GFED_regions = GFED_regions.rename({'lat' : 'latitude', 'lon' : 'longitude'})
transcom_regions = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/transcom_regions/transcom_regions_360_180.nc').transcom_regions
transcom_regions = transcom_regions.reindex(latitude=transcom_regions.latitude[::-1])
transcom_regions = transcom_regions.where(transcom_regions<=11)
transcom_regions = transcom_regions.where((transcom_regions<5) | (transcom_regions>6) )
transcom_regions = transcom_regions.where(np.isfinite(transcom_regions), GFED_regions)
transcom_mask ={"class_1":{"eco_class":  1, "name": "NA Bor."},
                "class_2":{"eco_class" : 2, "name": "NA Temp."},
                "class_3":{"eco_class" : 3, "name": "SA Trop."},
                "class_4":{"eco_class" : 4, "name": "SA Temp."},
                "class_5":{"eco_class" : 5, "name": "N Africa" },
                "class_6":{"eco_class" : 6, "name": "S Africa"},
                "class_7":{"eco_class" : 7, "name": "Eurasia Bor."},
                "class_8":{"eco_class" : 8, "name": "Eurasia Temp."},
                "class_9":{"eco_class" : 9, "name": "Trop. Asia"},
                "class_10":{"eco_class" : 10, "name": "Australia"},
                "class_11":{"eco_class" : 11, "name": "Europe"}}

fig, ax = plt.subplots(1,1, figsize=(5, 4), constrained_layout=True)

projection = ccrs.Robinson()
transcom_regions.plot.imshow(ax=ax, add_colorbar=False, cmap='tab20b')

# Set the 'bad' color in the colormap to grey for NaN values
#plt.cm.viridis.set_bad(color='grey')

# Annotate regions
# You will need the coordinates for each class. Here's an example:
for class_key, class_info in transcom_mask.items():
    lat, lon = get_coordinates_of_class_center(transcom_regions, class_info['eco_class'])
    ax.text(lon, lat, class_info['name'], ha='center', fontsize= 8, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('latitude [degrees north]', size=10)
ax.set_xlabel('longitude [degrees east]', size=10)
ax.set_title('')

plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS3.png', dpi=300)
