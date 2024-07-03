#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:06:30 2023

@author: simon
"""
import xarray as xr
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer
from pystac.extensions.eo import EOExtension as eo
import odc.stac
import matplotlib.gridspec as gridspec

# Function to load and plot satellite data
def load_and_plot_satellite_data(bbox, ax, bands=["B02", "B03", "B04"], collection="sentinel-2-l2a", time_of_interest="2020-01-01/2020-12-31"):
    search = catalog.search(collections=[collection], bbox=bbox, datetime=time_of_interest, query={"eo:cloud_cover": {"lt": 10}})
    items = search.item_collection()
    selected_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

    data = odc.stac.stac_load([selected_item], bands=bands, bbox=bbox,
                              crs="EPSG:4326",   resolution=0.00009009).isel(time=0)
    data[["B04", "B03", "B02"]].to_array().plot.imshow(robust=True, ax=ax)

# Function to calculate and plot forest age difference
def plot_forest_age_diff(ds, lat_range, lon_range, time1, time2, ax, cmap="bwr_r", vmin=-10, vmax=10):
    dat_ = ds.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
    age_diff_ = dat_.sel(time=time1) - dat_.sel(time=time2)
    #age_diff_ = age_diff_.where(age_diff_ != 0)
    cmap = plt.cm.bwr_r  # Replace 'viridis' with your colormap
    cmap.set_bad(color='grey')
    age_diff_.attrs['long_name'] = 'Forest Age [years]'
    age_diff_.plot.imshow(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                          cbar_kwargs=dict(orientation='vertical', shrink=0.6, aspect=10, pad=0.05))

# Initialize catalog
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)

# Load forest age dataset
ds = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/AgeUpscale_100m').forest_age

# Set up plot
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), constrained_layout=True)

# Define regions of interest
regions = {
    "Cost Range, USA - Forest harvest and regrowth": ([-123.4, 43.7, -123.3, 43.8], [43.8, 43.7], [-123.4, -123.3], "2020-01-01", "2010-01-01"),
    #"Southern Rookies, USA - Wildfire": ([-108.3, 37.3, -108.2, 37.4], [37.4, 37.3], [-108.3, -108.2], "2020-01-01", "2010-01-01"),
    #"Les Landes, FR - Wildfire": ([0.1, 43.9, 0.2, 44], [0.1, 0.2], [43.9, 44], "2020-01-01", "2010-01-01"),
    "Scandinavian Peninsula - Plantation": ([14.8, 60.2, 14.9, 60.3], [60.3, 60.2], [14.8, 14.9], "2020-01-01", "2010-01-01"),
    "Amazon, BR - Secondary forest regrowth": ([-52.3, -3, -52.2, -2.9], [-2.9, -3], [-52.3, -52.2], "2020-01-01", "2010-01-01"),
    #"Central African Forests, Congo Basin, BR - Selective logging": ([24.3, 0.18, 24.7, 0.21], [0.18, 0.21], [24.3, 24.7], "2020-01-01", "2010-01-01"),
                                        
}

# Iterate over regions and plot data
fig = plt.figure(figsize=(12, 6 * len(regions)),constrained_layout= True)

# Create a GridSpec for the entire figure
outer_grid = gridspec.GridSpec(len(regions), 1, figure=fig)

# Iterate over regions
for i, (region, (bbox, lat_range, lon_range, time1, time2)) in enumerate(regions.items()):
    # Create a nested 1x2 grid for each region within the outer grid
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i])

    # Create subplots
    ax1 = fig.add_subplot(inner_grid[0])
    ax2 = fig.add_subplot(inner_grid[1])

    # Plot satellite data in the first subplot
    load_and_plot_satellite_data(bbox, ax1)
    
    # Plot forest age difference in the second subplot
    plot_forest_age_diff(ds, lat_range, lon_range, time1, time2, ax2)

    # Set titles and aspect ratio
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax1.set_title('', fontweight='bold')
    
    # Set x-axis and y-axis label
    ax1.set_xlabel('longitude [degrees east]')
    ax2.set_xlabel('longitude [degrees east]')
    ax1.set_ylabel('latitude [degrees east]')
    ax2.set_ylabel('latitude [degrees east]')
    
    ax_middle = fig.add_subplot(outer_grid[i], frame_on=False)
    ax_middle.set_xticks([])
    ax_middle.set_yticks([])
    ax_middle.set_title(f'{region}', fontweight='bold', fontsize=18)
    
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS1.png', dpi=300)
