# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de

import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np

# Set matplotlib parameters for consistent styling
params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
    'font.size': 16,
    # axes
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 14,
    # tex
    'text.usetex': True,
}

mpl.rcParams.update(params)

# Directories
data_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/data/Age_upscale_100m'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

# Load data
forest_fraction = xr.open_zarr(os.path.join(data_dir, 'ForestFraction_1deg')).forest_fraction
age_2020 = xr.open_zarr(os.path.join(data_dir, 'AgeClass_1deg')).sel(time='2020-01-01').median(dim='members')
age_2020 = age_2020.where(forest_fraction > 0.2)

# Reclassify '200-299' and '>299' into '>200'
reclass_map = {'200-299': '>200', '>299': '>200'}
original = age_2020['forest_age']
keep_classes = [cls for cls in original.age_class.values if cls not in reclass_map]
new_classes = sorted(set(keep_classes + list(reclass_map.values())), key=lambda x: int(x.split('-')[0].replace('>', '999')))

reclassified = xr.DataArray(
    np.zeros((len(new_classes), original.sizes['latitude'], original.sizes['longitude']), dtype=np.float32),
    dims=('age_class', 'latitude', 'longitude'),
    coords={'age_class': new_classes, 'latitude': original.latitude, 'longitude': original.longitude},
    name='forest_age'
)

for cls in original.age_class.values:
    target = reclass_map.get(cls, cls)
    reclassified.loc[dict(age_class=target)] += original.sel(age_class=cls).values

# Shift age classes for scenario
shift_map = {
    '0-20': '20-40', '20-40': '40-60', '40-60': '60-80', '60-80': '80-100',
    '80-100': '100-120', '100-120': '120-140', '120-140': '140-160', '140-160': '160-180',
    '160-180': '180-200', '180-200': '>200', '>200': '>200' 
}

final_classes = sorted(set(shift_map.values()), key=lambda x: int(x.split('-')[0].replace('>', '999')))
age_2060 = xr.DataArray(
    np.zeros((len(final_classes), len(original.latitude), len(original.longitude)), dtype=np.float32),
    dims=('age_class', 'latitude', 'longitude'),
    coords={'age_class': final_classes, 'latitude': original.latitude, 'longitude': original.longitude},
    name='forest_age'
).to_dataset()

for old_cls, new_cls in shift_map.items():
    if old_cls in reclassified.age_class:
        age_2060['forest_age'].loc[dict(age_class=new_cls)] += reclassified.sel(age_class=old_cls)

# Midpoints
midpoints = {
    '0-20': 10, '20-40': 30, '40-60': 50, '60-80': 70, '80-100': 90,
    '100-120': 110, '120-140': 130, '140-160': 150, '160-180': 170,
    '180-200': 190, '>200': 200
}

# Compute weighted mean
weighted_2020 = sum(reclassified.sel(age_class=k) * v for k, v in midpoints.items() if k in reclassified.age_class)
weighted_2060 = sum(age_2060['forest_age'].sel(age_class=k) * v for k, v in midpoints.items() if k in age_2060.age_class)

weighted_2020 = weighted_2020.where(weighted_2020 > 0)
weighted_2060 = weighted_2060.where(weighted_2060 > 0)

# Plotting
fig, ax = plt.subplots(2, 2, subplot_kw={'projection': ccrs.Robinson()}, figsize=(9, 7), constrained_layout=True)
cbar_opts = dict(orientation='horizontal', shrink=0.8, aspect=40, pad=0.05, spacing='proportional', label='Forest age [years]')

weighted_2020.plot.imshow(ax=ax[0, 0], transform=ccrs.PlateCarree(), cmap='gist_earth_r', cbar_kwargs=cbar_opts)
ax[0, 0].coastlines(); ax[0, 0].gridlines(); ax[0, 0].set_title('Forest age in 2050\nunder BAU scenario')
ax[0, 0].text(0.05, 1.05, '(a)', transform=ax[0, 0].transAxes, fontsize=16, fontweight='bold', va='top')

weighted_2060.plot.imshow(ax=ax[0, 1], transform=ccrs.PlateCarree(), cmap='gist_earth_r', cbar_kwargs=cbar_opts)
ax[0, 1].coastlines(); ax[0, 1].gridlines(); ax[0, 1].set_title('Forest age in 2060\nunder conservation scenario')
ax[0, 1].text(0.05, 1.05, '(b)', transform=ax[0, 1].transAxes, fontsize=16, fontweight='bold', va='top')

diff = (weighted_2060 - weighted_2020).where((weighted_2060 > 0) & (weighted_2020 > 0))
diff.plot.imshow(ax=ax[1, 1], transform=ccrs.PlateCarree(), cmap='Blues', cbar_kwargs=cbar_opts)
ax[1, 1].coastlines(); ax[1, 1].gridlines(); ax[1, 1].set_title('Conservation - BAU')
ax[1, 1].text(0.05, 1.05, '(c)', transform=ax[1, 1].transAxes, fontsize=16, fontweight='bold', va='top')

fig.delaxes(ax[1, 0])
plt.savefig(os.path.join(plot_dir, 'figS5.png'), dpi=300)
