#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
"""
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
from ageUpscaling.utils.plotting import get_coordinates_of_class_center

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

#%% Specify data and plot directories
data_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/data/Age_upscale_100m'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

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
plt.savefig(os.path.join(plot_dir,'figS3.png'), dpi=300)
