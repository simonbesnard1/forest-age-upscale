import xarray as xr
import numpy as np
import pandas as pd


#%% Load forest fraction
forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction

#%% These should be replaced with your actual age data arrays/matrices for 2010 and 2020
out = []
for member_ in np.arange(20):
    
    #%%Load AGB changes
    BiomassDiffPartition_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/BiomassDiffPartition_1deg').sel(members = member_).stand_replaced
    
    #%%Load AGB 
    BiomassPartition_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/BiomassPartition_1deg').sel(members = member_)
    
    #%% Load stand-replace age class data
    AgeDiffPartition_fraction_1deg =  xr.open_zarr("/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiffPartition_1deg").sel(members = member_)

    
    #%% Calculate pixel area
    EARTH_RADIUS = 6371.0
    delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    width_of_longitude = EARTH_RADIUS * delta_lon
    delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    height_of_latitude = EARTH_RADIUS * delta_lat
    pixel_area = (width_of_longitude * height_of_latitude *
                  np.cos(np.deg2rad(forest_fraction.latitude))).broadcast_like(forest_fraction) * 1000000

    # Initialize a dictionary to hold the total area for each age class
    total_AGB_changes_stand_replaced = {}
    total_AGB_stand_replaced = {}
    total_AGB_aging = {}
    
    # Iterate over each age class, calculate the total area, and store it in the dictionary
    for age_class in AgeDiffPartition_fraction_1deg.age_class.values:
        AGBchange_stand_replaced =  (BiomassDiffPartition_1deg.sel(age_class= age_class) * 0.5 *-1 *100)/ 10
        AGB_stand_replaced =  BiomassPartition_1deg.stand_replaced.sel(age_class= age_class) * 0.5
        AGB_aging =  BiomassPartition_1deg.gradually_ageing.sel(age_class= age_class) * 0.5
        
        Fraction_stand_replaced =  AgeDiffPartition_fraction_1deg.sel(age_class = age_class).stand_replaced_class_partition
        Fraction_stand_replaced = Fraction_stand_replaced.where(Fraction_stand_replaced >0)
        Fraction_aging =  AgeDiffPartition_fraction_1deg.sel(age_class = age_class).aging_forest_class_partition
        Fraction_aging = Fraction_aging.where(Fraction_stand_replaced >0)
        
        AGB_stand_replaced_total = AGBchange_stand_replaced * pixel_area * forest_fraction * Fraction_stand_replaced.values 
        total_budget_stand_replaced = np.nansum(AGB_stand_replaced_total)
        total_budget_stand_replaced = np.nansum(AGB_stand_replaced_total) * 1e-15
        
        # Multiply the age fraction by the pixel area and sum over all pixels
        total_AGB_changes_stand_replaced[age_class] = total_budget_stand_replaced
        
        stand_replaced_total = AGB_stand_replaced * pixel_area * forest_fraction * Fraction_stand_replaced.values 
        total_AGB_stand_replaced_class = np.nansum(stand_replaced_total)
        total_AGB_stand_replaced_class = np.nansum(total_AGB_stand_replaced_class) * 1e-13
        total_AGB_stand_replaced[age_class] = total_AGB_stand_replaced_class
        
        aging_total = AGB_aging * pixel_area * forest_fraction * Fraction_aging.values 
        total_AGB_aging_class = np.nansum(aging_total)
        total_AGB_aging_class = np.nansum(total_AGB_aging_class) * 1e-13
        total_AGB_aging[age_class] = total_AGB_aging_class
        
    # Creating a DataFrame
    df = pd.DataFrame({
        'member': member_,
        'age_class': list(total_AGB_changes_stand_replaced.keys()),
        'AGBchanges_stand_replaced': list(total_AGB_changes_stand_replaced.values()),
        'AGB_stand_replaced': list(total_AGB_stand_replaced.values()),
        'AGB_aging': list(total_AGB_aging.values()),
        
    })
    
    total_AGBchanges_stand_replaced = df['AGBchanges_stand_replaced'].sum()
    total_AGB_stand_replaced = df['AGB_stand_replaced'].sum()
    total_AGB_aging = df['AGB_aging'].sum()
    new_row = pd.DataFrame({'member': [member_], 'age_class': ['all_class'], 'AGBchanges_stand_replaced': [total_AGBchanges_stand_replaced], 'AGB_stand_replaced': [total_AGB_stand_replaced],'AGB_aging': [total_AGB_aging]})
    df = pd.concat([df, new_row], ignore_index=True)
    out.append(df)
    
#%% Compute statistics
out= pd.concat(out)
median_out = out.groupby("age_class").median(numeric_only=True)
q5_out = out.groupby("age_class").quantile(numeric_only=True, q=0.05)
q95_out = out.groupby("age_class").quantile(numeric_only=True, q=0.95)

