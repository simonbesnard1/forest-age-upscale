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

#%% load library
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ageUpscaling.utils.plotting import calculate_pixel_area
import os

# Set matplotlib parameters for consistent styling
params = {
    # font
    'font.family': 'serif',
    'font.size': 14,
    # axes
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 12
}

mpl.rcParams.update(params)

#%% Specify data and plot directories
data_dir = '/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

#%% Load stand-replace age class data
forest_fraction = forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction

#%% Load stand-replace age class data
AgeDiffPartition_fraction_1deg =  xr.open_zarr(os.path.join(data_dir,"AgeDiffPartition_fraction_1deg"))

Young_stand_replaced_class =  AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '0-20')
Young_stand_replaced_class = Young_stand_replaced_class.where(Young_stand_replaced_class >0)

Intermediate_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = ['20-40', '40-60', '60-80']).sum(dim='age_class')
Intermediate_stand_replaced_class = Intermediate_stand_replaced_class.where(Intermediate_stand_replaced_class >0)

Mature_stand_replaced_class =  AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = ['80-100', '100-120', '120-140', '140-160', '160-180', '180-200']).sum(dim='age_class')
Mature_stand_replaced_class = Mature_stand_replaced_class.where(Mature_stand_replaced_class >0)

OG_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '>200')
OG_stand_replaced_class = OG_stand_replaced_class.where(OG_stand_replaced_class >0)

Young_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '0-20')
Young_aging_class = Young_aging_class.where(Young_aging_class >0)

Intermediate_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = ['20-40', '40-60', '60-80']).sum(dim='age_class')
Intermediate_aging_class = Intermediate_aging_class.where(Intermediate_aging_class >0)

Mature_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = ['80-100', '100-120', '120-140', '140-160', '160-180', '180-200']).sum(dim='age_class')
Mature_aging_class = Mature_aging_class.where(Mature_aging_class >0)

OG_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '>200')
OG_aging_class = OG_aging_class.where(OG_aging_class >0)

#%% Calculate pixel area
pixel_area = calculate_pixel_area(forest_fraction, 
                                  EARTH_RADIUS = 6378.160, 
                                  resolution=1)

#%% Load transcom data
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
transcom_mask ={"class_7":{"eco_class" : 7, "name": "Eurasia Boreal"},                
                "class_1":{"eco_class":  1, "name": "NA Boreal"},
                "class_8":{"eco_class" : 8, "name": "Eurasia Temperate"},
                "class_11":{"eco_class" : 11, "name": "Europe"},                
                "class_2":{"eco_class" : 2, "name": "NA Temperate"},
                "class_4":{"eco_class" : 4, "name": "SA Temperate"},
                "class_3":{"eco_class" : 3, "name": "SA Tropical"},
                "class_9":{"eco_class" : 9, "name": "Tropical Asia"},
                "class_5":{"eco_class" : 5, "name": "Northern Africa"},
                "class_6":{"eco_class" : 6, "name": "Southern Africa"},
                "class_10":{"eco_class" : 10, "name": "Australia"}}

#%% Compute total area per age class and NEE for each transcom regions.
total_area_Young_stand_replaced_class = {}
total_area_Intermediate_stand_replaced_class = {}
total_area_Mature_stand_replaced_class = {}
total_area_OG_stand_replaced_class = {}
total_area_Young_aging_class = {}
total_area_Intermediate_aging_class = {}
total_area_Mature_aging_class = {}
total_area_OG_aging_class = {}
total_area_forest = {}

for region_ in list(transcom_mask.keys()):
    class_values = transcom_mask[region_]['eco_class']
    class_name = transcom_mask[region_]['name']
    total_area_Young_stand_replaced_class[class_name] = (Young_stand_replaced_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_Intermediate_stand_replaced_class[class_name] = (Intermediate_stand_replaced_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_Mature_stand_replaced_class[class_name] = (Mature_stand_replaced_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_OG_stand_replaced_class[class_name] = (OG_stand_replaced_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_Young_aging_class[class_name] = (Young_aging_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_Intermediate_aging_class[class_name] = (Intermediate_aging_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_Mature_aging_class[class_name] = (Mature_aging_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_OG_aging_class[class_name] = (OG_aging_class.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_forest[class_name] = (forest_fraction.where(transcom_regions==class_values) * pixel_area).sum(dim=['latitude', 'longitude']).values / 10**7
   
#%% Plot
width = 0.2  # Reduce the width of the bars
fig, ax = plt.subplots(1, 2, figsize=(10.5, 6), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

x = np.arange(len(total_area_Young_stand_replaced_class))  # the label locations
ax[0].bar(x, list(total_area_Young_stand_replaced_class.values()), label='Young stand-replaced', color='#018571')
ax[0].bar(x, list(total_area_Intermediate_stand_replaced_class.values()), bottom=list(total_area_Young_stand_replaced_class.values()), label='Maturing stand-replaced', color='#80cdc1')
bottom_for_mature = np.add(list(total_area_Young_stand_replaced_class.values()), list(total_area_Intermediate_stand_replaced_class.values()))
ax[0].bar(x, list(total_area_Mature_stand_replaced_class.values()), bottom=bottom_for_mature, label='Mature stand-replaced', color='#dfc27d')
bottom_for_og = np.add(bottom_for_mature, list(total_area_Mature_stand_replaced_class.values()))
ax[0].bar(x, list(total_area_OG_stand_replaced_class.values()), bottom=bottom_for_og, label='Old-growth stand-replaced', color='#a6611a')

ax[0].set_xticks(x)
ax[0].set_xticklabels(list(total_area_Young_stand_replaced_class.keys()), rotation=90)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylabel('Area [billion hectares]', size=12)
ax[0].legend(frameon=False, fontsize=7.8, loc='upper left')
ax[0].text(0.01, 1.1, 'a', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top')


x = np.arange(len(total_area_Young_aging_class))  # the label locations
ax[1].bar(x, list(total_area_Young_aging_class.values()), label='Young ageing', color='#018571')
ax[1].bar(x, list(total_area_Intermediate_aging_class.values()), bottom=list(total_area_Young_aging_class.values()), label='Maturing ageing', color='#80cdc1')
bottom_for_mature = np.add(list(total_area_Young_aging_class.values()), list(total_area_Intermediate_aging_class.values()))
ax[1].bar(x, list(total_area_Mature_aging_class.values()), bottom=bottom_for_mature, label='Mature ageing', color='#dfc27d')
bottom_for_og = np.add(bottom_for_mature, list(total_area_Mature_aging_class.values()))
ax[1].bar(x, list(total_area_OG_aging_class.values()), bottom=bottom_for_og, label='Old-growth ageing', color='#a6611a')

ax[1].set_xticks(x)
ax[1].set_xticklabels(list(total_area_Young_aging_class.keys()), rotation=90)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylabel('Area [billion hectares]', size=12)
ax[1].legend(frameon=False, fontsize=7.8, loc='upper right')
ax[1].text(0.01, 1.1, 'b', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top')
plt.savefig(os.path.join(plot_dir,'figS7.png'), dpi=300)

#%% Calculate total stats
total_forest_area = np.array(list(total_area_forest.values()))
sum_loss =  np.sum(np.stack([list(total_area_Young_stand_replaced_class.values()), list(total_area_Intermediate_stand_replaced_class.values()), list(total_area_Mature_stand_replaced_class.values()), list(total_area_OG_stand_replaced_class.values())]), axis=0)
total_sum_loss = np.sum(sum_loss) / np.sum(total_forest_area)
relative_total_sum_loss = total_sum_loss *100

sum_aging = np.sum(np.stack([list(total_area_Young_aging_class.values()), list(total_area_Intermediate_aging_class.values()), list(total_area_Mature_aging_class.values()), list(total_area_OG_aging_class.values())]), axis=0)
total_sum_aging = np.sum(sum_aging) / np.sum(total_forest_area)
relative_total_sum_aging = total_sum_aging *100

#%% Calculate total stats per age class
total_forest_area = np.array(list(total_area_forest.values()))
sum_loss_young =  np.sum(list(total_area_Young_stand_replaced_class.values()))
total_sum_loss_young = sum_loss_young / np.sum(sum_loss)
relative_total_sum_loss_young = total_sum_loss_young *100

sum_loss_maturing =  np.sum(list(total_area_Intermediate_stand_replaced_class.values()))
total_sum_loss_maturing = sum_loss_maturing / np.sum(sum_loss)
relative_total_sum_loss_maturing = total_sum_loss_maturing *100

sum_loss_mature =  np.sum(list(total_area_Mature_stand_replaced_class.values()))
total_sum_loss_mature = sum_loss_mature / np.sum(sum_loss)
relative_total_sum_loss_mature = total_sum_loss_mature *100

sum_loss_OG =  np.sum(list(total_area_OG_stand_replaced_class.values()))
total_sum_loss_OG = sum_loss_OG / np.sum(sum_loss)
relative_total_sum_loss_OG = total_sum_loss_OG *100

sum_aging_young =  np.sum(list(total_area_Young_aging_class.values()))
total_sum_aging_young = sum_aging_young / np.sum(sum_aging)
relative_total_sum_aging_young = total_sum_aging_young *100

sum_aging_maturing =  np.sum(list(total_area_Intermediate_aging_class.values()))
total_sum_aging_maturing = sum_aging_maturing / np.sum(sum_aging)
relative_total_sum_aging_maturing = total_sum_aging_maturing *100

sum_aging_mature =  np.sum(list(total_area_Mature_aging_class.values()))
total_sum_aging_mature = sum_aging_mature / np.sum(sum_aging)
relative_total_sum_aging_mature = total_sum_aging_mature *100

sum_aging_OG =  np.sum(list(total_area_OG_aging_class.values()))
total_sum_aging_OG = sum_aging_OG / np.sum(sum_aging)
relative_total_sum_aging_OG = total_sum_aging_OG *100



