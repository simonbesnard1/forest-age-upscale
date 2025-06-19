#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
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
    'legend.fontsize': 12,
    'text.usetex': True,
}

mpl.rcParams.update(params)

#%% Specify data and plot directories
data_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/data/Age_upscale_100m'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

#%% These should be replaced with your actual age data arrays/matrices for 2010 and 2020
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction

summarized_areas_2020_members = []
summarized_areas_scenario1_members = []
AgePartition_total_scenario1 = []
AgePartition_total_BAU = [] 
for member_ in np.arange(20):

    age_2020 = xr.open_zarr(os.path.join(data_dir,'AgeClass_1deg')).sel(time = '2020-01-01').sel(members= member_).transpose("latitude", "longitude", "age_class").forest_age.where(forest_fraction>.2)
    # Sum the '200-299' and '>299' age classes into a new '>200' age class
    combined = age_2020.sel(age_class=['200-299', '>299']).sum(dim='age_class')
    
    # Remove the old age classes and add the new combined age class
    age_2020 = age_2020.drop_sel(age_class=['200-299'])
    age_2020 = age_2020.assign_coords(age_class=[ac if ac != '>299' else '>200' for ac in age_2020.age_class.values])
    age_2020.loc[:, :, '>200'] = combined
    
    # Sort the age_class dimension if needed
    age_2020 = age_2020.sortby('age_class')
    #%% Experiment with conservation scenario
    age_class_mapping = {
        '0-20': '20-40',
        '20-40': '40-60',
        '40-60': '60-80',
        '60-80': '80-100',
        '80-100': '100-120',
        '100-120': '120-140',
        '120-140': '140-160',
        '140-160': '160-180',
        '160-180': '180-200',  # Assuming >180 becomes >200
        '180-200': '>200',  # Assuming >180 becomes >200
        '>200': '>200'      # Assuming >200 remains >200
    }
    # Create a list of new age classes
    new_age_classes = np.concatenate([['0-20'], np.array(list(set(age_class_mapping.values())))])
    
    # Create age_2060_scenario1 with the new age classes
    age_2060_scenario1 = xr.zeros_like(age_2020).reindex(age_class=new_age_classes, fill_value=0)
    age_2060_scenario2 = xr.zeros_like(age_2020).reindex(age_class=new_age_classes, fill_value=0)

    for old_age_class, new_age_class in age_class_mapping.items():
        age_2060_scenario1.loc[:,:,new_age_class] += age_2020.loc[:,:,old_age_class]
        
    #%% Experiment with BAU scenario
    
    # Earth's radius in kilometers
    EARTH_RADIUS = 6371.0
    
    # Calculate the width of each longitude slice in radians
    # Multiply by Earth's radius to get the width in kilometers
    delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    width_of_longitude = EARTH_RADIUS * delta_lon
    
    # Calculate the height of each latitude slice in radians
    # Multiply by Earth's radius to get the height in kilometers
    delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    height_of_latitude = EARTH_RADIUS * delta_lat
    
    # Now, calculate the area of each pixel in square kilometers
    # cos(latitude) factor accounts for the convergence of meridians at the poles
    # We need to convert latitude from degrees to radians first
    pixel_area = (width_of_longitude * height_of_latitude *
                  np.cos(np.deg2rad(age_2020.latitude))).broadcast_like(age_2020.isel(age_class=0))
    
    # Initialize a dictionary to hold the total area for each age class
    total_area_per_age_class_2020 = {}
    total_area_per_age_class_scenario1 = {}
    
    # Iterate over each age class, calculate the total area, and store it in the dictionary
    for age_class in age_2020.age_class.values:
        # Multiply the age fraction by the pixel area and sum over all pixels
        total_area_per_age_class_scenario1[age_class] = (age_2060_scenario1.sel(age_class = age_class) * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
        total_area_per_age_class_2020[age_class] = (age_2020.sel(age_class = age_class) * pixel_area * forest_fraction).sum(dim=['longitude', 'latitude']).values / 10**7
        
    
    # Example age classes and counts for 2010 and 2020
    age_classes = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140-160', '160-180', '180-200', '>200']
    
    # These should be your actual counts of forest areas for each age class in 2010 and 2020.
    counts_2010 = np.stack(list(total_area_per_age_class_scenario1.values()))
    counts_2020 = np.stack(list(total_area_per_age_class_2020.values()))
    
    #%% Calculate area per forest type
    class_ranges = {
        'Young forests \n (0-20 years)': range(0, 21),   # Young Forests: 1-20 years (considering age 0 as 1)
        'Maturing forests \n (21-80 years)': range(21, 81),  # Maturing Forests: 21-80 years
        'Mature forests \n (81-200 years)': range(81, 201), # Mature Forests: 81-150 years
        'Old forests \n ($>$200 years)': range(201, 301) # Old-Growth Forests: 151-300 years
    }
    
    # Initialize a dictionary to hold the summarized areas
    summarized_areas_2020 = {
        'Young forests \n (0-20 years)': 0.0,
        'Maturing forests \n (21-80 years)': 0.0,
        'Mature forests \n (81-200 years)': 0.0,
        'Old forests \n ($>$200 years)': 0.0
    }
    
    summarized_areas_scenario1 = {
        'Young forests \n (0-20 years)': 0.0,
        'Maturing forests \n (21-80 years)': 0.0,
        'Mature forests \n (81-200 years)': 0.0,
        'Old forests \n ($>$200 years)': 0.0
    }
    
    # Function to determine the class of an age group based on its range
    def get_age_class(age_group):
        for class_name, age_range in class_ranges.items():
            if age_group in age_range:
                return class_name
        return None
        
    # Aggregate areas into the summarized areas dictionary
    for age_group, area in total_area_per_age_class_2020.items():
        # Extract the lower bound of the age group (e.g., 'age_0_10' -> 0)
        if age_group == '>200':
            lower_bound = 201
        else:
            lower_bound = int(age_group.split('-')[1])
        
        # Determine the class of the current age group
        age_class = get_age_class(lower_bound)
        # Add the area to the appropriate class if it's not None
        if age_class:
            summarized_areas_2020[age_class] += area
    
    for age_group, area in total_area_per_age_class_scenario1.items():
        # Extract the lower bound of the age group (e.g., 'age_0_10' -> 0)
        if age_group == '>200':
            lower_bound = 201
        else:
            lower_bound = int(age_group.split('-')[1])
            
        # Determine the class of the current age group
        age_class = get_age_class(lower_bound)
        # Add the area to the appropriate class if it's not None
        if age_class:
            summarized_areas_scenario1[age_class] += area


    #%% Load stand-replaced / aging agb difference
    BiomassDiffPartition_1deg =  xr.open_zarr(os.path.join(data_dir,'BiomassPartition_1deg')).sel(members = member_)
    Young_stand_replaced =  (BiomassDiffPartition_1deg.stand_replaced.where(forest_fraction>.2).sel(age_class= '0-20') * 0.47)
    Intermediate_stand_replaced = (BiomassDiffPartition_1deg.stand_replaced.where(forest_fraction>.2).sel(age_class= '20-80') * 0.47)
    Mature_stand_replaced =   (BiomassDiffPartition_1deg.stand_replaced.where(forest_fraction>.2).sel(age_class= '80-200') * 0.47)
    Old_stand_replaced =   (BiomassDiffPartition_1deg.stand_replaced.where(forest_fraction>.2).sel(age_class= '>200')* 0.47)
    Young_aging =  (BiomassDiffPartition_1deg.gradually_ageing.where(forest_fraction>.2).where(forest_fraction>.2).sel(age_class= '0-20') * 0.47) 
    Intermediate_aging= (BiomassDiffPartition_1deg.gradually_ageing.where(forest_fraction>.2).sel(age_class= '20-80') * 0.47)
    Mature_aging =   (BiomassDiffPartition_1deg.gradually_ageing.where(forest_fraction>.2).sel(age_class= '80-200') * 0.47)
    Old_aging =   (BiomassDiffPartition_1deg.gradually_ageing.where(forest_fraction>.2).sel(age_class= '>200')* 0.47)
    
    #%% Calculate pixel area
    EARTH_RADIUS = 6371.0
    delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    width_of_longitude = EARTH_RADIUS * delta_lon
    delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    height_of_latitude = EARTH_RADIUS * delta_lat
    pixel_area = (width_of_longitude * height_of_latitude *
                  np.cos(np.deg2rad(Intermediate_stand_replaced.latitude))).broadcast_like(Intermediate_stand_replaced) * 1000000
    
    #%% Load stand-replace age class data
    AgeDiffPartition_fraction_1deg =  xr.open_zarr(os.path.join(data_dir,"AgeDiffPartition_1deg")).sel(members = member_)
    
    Young_stand_replaced_class =  AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.where(forest_fraction>.2).sel(age_class = '0-20')
    Young_stand_replaced_class = Young_stand_replaced_class.where(Young_stand_replaced_class >0)
    
    Intermediate_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.where(forest_fraction>.2).sel(age_class = '20-80')
    Intermediate_stand_replaced_class = Intermediate_stand_replaced_class.where(Intermediate_stand_replaced_class >0)
    
    Mature_stand_replaced_class =  AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.where(forest_fraction>.2).sel(age_class = '80-200')
    Mature_stand_replaced_class = Mature_stand_replaced_class.where(Mature_stand_replaced_class >0)
    
    Old_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.where(forest_fraction>.2).sel(age_class = '>200')
    Old_stand_replaced_class = Old_stand_replaced_class.where(Old_stand_replaced_class >0)
    
    Young_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.where(forest_fraction>.2).sel(age_class = '0-20')
    Young_aging_class = Young_aging_class.where(Young_aging_class >0)
    
    Intermediate_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.where(forest_fraction>.2).sel(age_class = '20-80')
    Intermediate_aging_class = Intermediate_aging_class.where(Intermediate_aging_class >0)
    
    Mature_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.where(forest_fraction>.2).sel(age_class = '80-200')
    Mature_aging_class = Mature_aging_class.where(Mature_aging_class >0)
    
    Old_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.where(forest_fraction>.2).sel(age_class = '>200')
    Old_aging_class = Old_aging_class.where(Old_aging_class >0)
    
    #%% Calculate total per area
    Young_stand_replaced_total = Young_stand_replaced * pixel_area * forest_fraction * Young_stand_replaced_class
    Intermediate_stand_replaced_total = Intermediate_stand_replaced *  pixel_area * forest_fraction  * Intermediate_stand_replaced_class
    Mature_stand_replaced_total = Mature_stand_replaced * pixel_area * forest_fraction  * Mature_stand_replaced_class
    Old_stand_replaced_total = Old_stand_replaced * pixel_area * forest_fraction  * Old_stand_replaced_class.values
    Young_aging_total = Young_aging * pixel_area * forest_fraction * Young_aging_class.values 
    Intermediate_aging_total = Intermediate_aging * pixel_area * forest_fraction * Intermediate_aging_class.values 
    Mature_aging_total = Mature_aging * pixel_area * forest_fraction * Mature_aging_class.values 
    Old_aging_total = Old_aging * pixel_area * forest_fraction * Old_aging_class.values 
    
    #%% Compute total budget - BAU
    total_budget_Young_stand_replaced = np.nansum(Young_stand_replaced_total)
    total_budget_Young_stand_replaced = np.nansum(total_budget_Young_stand_replaced) * 1e-13
    
    total_budget_Intermediate_stand_replaced = np.nansum(Intermediate_stand_replaced_total)
    total_budget_Intermediate_stand_replaced = np.nansum(total_budget_Intermediate_stand_replaced) * 1e-13
    
    total_budget_Mature_stand_replaced = np.nansum(Mature_stand_replaced_total)
    total_budget_Mature_stand_replaced = np.nansum(total_budget_Mature_stand_replaced) * 1e-13
    
    total_budget_Old_stand_replaced = np.nansum(Old_stand_replaced_total)
    total_budget_Old_stand_replaced = np.nansum(total_budget_Old_stand_replaced) * 1e-13
    
    total_budget_Young_aging = np.nansum(Young_aging_total)
    total_budget_Young_aging = np.nansum(total_budget_Young_aging) * 1e-13
    
    total_budget_Intermediate_aging = np.nansum(Intermediate_aging_total)
    total_budget_Intermediate_aging = np.nansum(total_budget_Intermediate_aging) * 1e-13
    
    total_budget_Mature_aging = np.nansum(Mature_aging_total)
    total_budget_Mature_aging = np.nansum(total_budget_Mature_aging) * 1e-13
    
    total_budget_Old_aging = np.nansum(Old_aging_total)
    total_budget_Old_aging = np.nansum(total_budget_Old_aging) * 1e-13
    
    total_young = total_budget_Young_stand_replaced +  total_budget_Intermediate_stand_replaced + total_budget_Mature_stand_replaced + total_budget_Young_aging  + total_budget_Old_stand_replaced
    
    total_budget_BAU = total_young + total_budget_Old_aging +  total_budget_Intermediate_aging + total_budget_Mature_aging
    
    #%% Compute total budget - forest conservation scenario
    total_budget_Young_scenario1 = (total_young * summarized_areas_scenario1['Young forests \n (0-20 years)']) /  summarized_areas_2020['Young forests \n (0-20 years)']
    
    total_budget_Intermediate_scenario1 = (total_budget_Intermediate_aging * summarized_areas_scenario1['Maturing forests \n (21-80 years)']) /  summarized_areas_2020['Maturing forests \n (21-80 years)']
    
    total_budget_Mature_scenario1 =  (total_budget_Mature_aging * summarized_areas_scenario1['Mature forests \n (81-200 years)']) /  summarized_areas_2020['Mature forests \n (81-200 years)']
    
    total_budget_Old_scenario1 = (total_budget_Old_aging * summarized_areas_scenario1['Old forests \n ($>$200 years)']) /  summarized_areas_2020['Old forests \n ($>$200 years)']
    
    total_budget_scenario1 = total_budget_Young_scenario1 + total_budget_Intermediate_scenario1 +  total_budget_Mature_scenario1 + total_budget_Old_scenario1

    #%% create final dataset
    summarized_areas_2020['member'] = member_
    summarized_areas_scenario1['member'] = member_
    summarized_areas_2020_members.append(pd.DataFrame([summarized_areas_2020]))
    summarized_areas_scenario1_members.append(pd.DataFrame([summarized_areas_scenario1]))

    
    AgePartition_total_scenario1.append(pd.DataFrame({'member': [member_], 'Young stand-replaced':[total_budget_Young_scenario1], 'Maturing stand-replaced':[total_budget_Intermediate_scenario1], 
                                                 'Mature stand-replaced':[total_budget_Mature_scenario1], 'Old stand-replaced':[total_budget_Old_scenario1], 'All stand-replaced':[total_budget_scenario1]}))

    AgePartition_total_BAU.append(pd.DataFrame({'member': [member_], 'Young stand-replaced':[total_young], 'Maturing stand-replaced':[total_budget_Intermediate_aging], 
                                           'Mature stand-replaced':[total_budget_Mature_aging],'Old stand-replaced':[total_budget_Old_aging], 'All stand-replaced':[total_budget_BAU]}))

    
AgePartition_total_BAU = pd.concat(AgePartition_total_BAU)
AgePartition_total_scenario1 = pd.concat(AgePartition_total_scenario1)
summarized_areas_2020_members = pd.concat(summarized_areas_2020_members)
summarized_areas_scenario1_members = pd.concat(summarized_areas_scenario1_members)

#%% Setup the figure and axes for a 2x2 grid
width = 0.3  # Reduce the width of the bars
fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

x = np.arange(len(list(class_ranges.keys())))  # the label locations

median_out = summarized_areas_2020_members.median(numeric_only=True)
q5_out = summarized_areas_2020_members.quantile(numeric_only=True, q=0.05)
q95_out = summarized_areas_2020_members.quantile(numeric_only=True, q=0.95)
error_bars_ = np.vstack((median_out.values[x] - q5_out.values[x],
                             q95_out.values[x] - median_out.values[x]))
bars1 = ax[1].bar(x - width/2, median_out.values[x], width, yerr=error_bars_, capsize=2,
                  label='Business-as-usual scenario', color='green')

median_out = summarized_areas_scenario1_members.median(numeric_only=True)
q5_out = summarized_areas_scenario1_members.quantile(numeric_only=True, q=0.05)
q95_out = summarized_areas_scenario1_members.quantile(numeric_only=True, q=0.95)
error_bars_ = np.vstack((median_out.values[x] - q5_out.values[x],
                             q95_out.values[x] - median_out.values[x]))

bars2 = ax[1].bar(x + width/2, median_out.values[x], width, yerr=error_bars_, capsize=2,
                  label='Forest conservation scenario', color='blue')

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylabel('Area [billion hectares]', size=14)
ax[1].set_xticks(x)
ax[1].set_xticklabels(list(class_ranges.keys()), size=14)

ax[1].legend(frameon=False, fontsize=16, bbox_to_anchor=(0.6,-0.2))
ax[1].text(0.05, 1.05, '(b)', transform=ax[1].transAxes,
            fontsize=18, fontweight='bold', va='top')
ax[1].set_title('Forest age class distribution in 2050', fontsize=16, fontweight='bold')

x = np.arange(len(np.concatenate([list(class_ranges.keys()) , ['All forests']])))  # the label locations
median_out = AgePartition_total_scenario1.median(numeric_only=True)
q5_out = AgePartition_total_scenario1.quantile(numeric_only=True, q=0.05)
q95_out = AgePartition_total_scenario1.quantile(numeric_only=True, q=0.95)
error_bars_ = np.vstack((median_out.values[x+1] - q5_out.values[x+1],
                             q95_out.values[x+1] -median_out.values[x+1]))
ax[0].bar(x + width/2, median_out.values[x+1], color='blue', width=width,
          yerr=error_bars_, capsize=2,)


median_out = AgePartition_total_BAU.median(numeric_only=True)
q5_out = AgePartition_total_BAU.quantile(numeric_only=True, q=0.05)
q95_out = AgePartition_total_BAU.quantile(numeric_only=True, q=0.95)
error_bars_ = np.vstack((median_out.values[x+1] - q5_out.values[x+1],
                             q95_out.values[x+1] -median_out.values[x+1]))

ax[0].bar(x - width/2, median_out.values[x+1], color='green', width=width,
          yerr=error_bars_, capsize=2,)
    
ax[0].set_xticks(x)
ax[0].set_xticklabels(list(np.concatenate([list(class_ranges.keys()) , ['All forests']])), size=14)
ax[0].set_ylabel('Carbon stock [PgC]', size=14)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].text(0.05, 1.05, '(a)', transform=ax[0].transAxes,
            fontsize=18, fontweight='bold', va='top')
ax[0].set_title('Total carbon stock in 2050', fontsize=16, fontweight='bold')
plt.savefig(os.path.join(plot_dir,'figExt9.png'), dpi=300)
