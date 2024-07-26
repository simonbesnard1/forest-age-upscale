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

#%% Load library
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib as mpl
import os 
from ageUpscaling.utils.plotting import filter_nan_gaussian_conserving

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

#%% Load management data
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction
management_fraction = xr.open_zarr(os.path.join(data_dir,'ManagementTypeFrac_1deg')).where(forest_fraction >0)

fig, ax = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

i= 0
for management_type in ['intact_forests', ['naturally_regenerated']]:
    
    #%% Load stand-replace age class data
    AgeDiff_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiff_1deg').median(dim = 'members')
    if management_type == 'intact_forests':
        AgeDiff_1deg = AgeDiff_1deg.where(management_fraction[[management_type]].to_array().sum(dim="variable")>0.5)
    else:
        AgeDiff_1deg = AgeDiff_1deg.where(management_fraction[management_type].to_array().sum(dim="variable")>0.5)
    
    total_aging_area =  AgeDiff_1deg.aging_forest_class
    total_stand_replaced_area = AgeDiff_1deg.stand_replaced_class
   
    #%% Calculate pixel area
    EARTH_RADIUS = 6371.0
    delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    width_of_longitude = EARTH_RADIUS * delta_lon
    delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
    height_of_latitude = EARTH_RADIUS * delta_lat
    pixel_area = (width_of_longitude * height_of_latitude *
                  np.cos(np.deg2rad(forest_fraction.latitude))).broadcast_like(forest_fraction) * 1000000
    
    #%% Load lateral flux data
    lateral_fluxes_sink = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[["allcropsink", "biofuelcropsink", 'riversink']]
    lateral_fluxes_source = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/LSCE_lateral_fluxes/lateralfluxes_2021_v4.1_1d00.nc')[['allcropsource', "biofuelcropsource", 'lakeriveremis']]
    lateral_fluxes_sink_2010 = lateral_fluxes_sink.sel(time= slice('2009-01-01', '2011-12-31')).mean(dim='time').to_array().sum(dim='variable')
    lateral_fluxes_sink_2020 = lateral_fluxes_sink.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')
    lateral_fluxes_source_2010 = lateral_fluxes_source.sel(time= slice('2009-01-01', '2011-12-31')).mean(dim='time').to_array().sum(dim='variable')
    lateral_fluxes_source_2020 = lateral_fluxes_source.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').to_array().sum(dim='variable')
    
    #%% Load NEE data
    land_fraction = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/inversion_jena/s99oc_v2022_daily/NEE.daily.360.180.2020.nc').LF
    NEE_RECCAP = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/GCP/GCP2023_inversions_1x1_version1_1_20240124.nc').land_flux_only_fossil_cement_adjusted
    
    nee_bin_members = []
    for member_ in NEE_RECCAP.ensemble_member:
        NEE_2010 =  NEE_RECCAP.sel(time= slice('2009-01-01', '2011-12-31')).isel(ensemble_member= member_).mean(dim='time') * 1e+15
        if not np.isnan(NEE_2010.values).all():
            NEE_2010 = NEE_2010 - lateral_fluxes_sink_2010 + lateral_fluxes_source_2010
            NEE_2010 = NEE_2010.where(forest_fraction>0)
            NEE_2010_filtered = filter_nan_gaussian_conserving(NEE_2010, length_km=500)
            NEE_2010_filtered = NEE_2010_filtered.reindex(latitude=NEE_2010_filtered.latitude[::-1])
            NEE_2010_filtered['latitude'] = forest_fraction['latitude']
            NEE_2010_filtered['longitude'] = forest_fraction['longitude']
            NEE_2010_filtered = NEE_2010_filtered.where(forest_fraction>0)
            if management_type == 'intact_forests':
                NEE_2010_filtered = NEE_2010_filtered.where(management_fraction[[management_type]].to_array().sum(dim="variable")>0.5)
            else:
                NEE_2010_filtered = NEE_2010_filtered.where(management_fraction[management_type].to_array().sum(dim="variable")>0.5)
            
            NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).isel(ensemble_member= member_).mean(dim='time') * 1e+15
            NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
            NEE_2020 = NEE_2020.where(forest_fraction>0)
            NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)
            NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
            NEE_2020_filtered['latitude'] = forest_fraction['latitude']
            NEE_2020_filtered['longitude'] = forest_fraction['longitude']
            NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>0)
            if management_type == 'intact_forests':
                NEE_2020_filtered = NEE_2020_filtered.where(management_fraction[[management_type]].to_array().sum(dim="variable")>0.5)
            else:
                NEE_2020_filtered = NEE_2020_filtered.where(management_fraction[management_type].to_array().sum(dim="variable")>0.5)
            
            NEE_diff_2020_2010 = (NEE_2020_filtered - NEE_2010_filtered) #*pixel_area / 1e12
            
            ratio_ = total_stand_replaced_area.values.reshape(-1) / total_aging_area.values.reshape(-1)
            nee_ = NEE_diff_2020_2010.values.reshape(-1)
            
            AgeBins = np.concatenate([np.arange(0, .6, 0.01), [1]])
            nee_bin = []
            for j in range(len(AgeBins)-1):
                Agemask = (ratio_ > AgeBins[j]) & (ratio_ <= AgeBins[j+1])
                NEE_masked = nee_[Agemask]
                nee_bin.append(np.nanmedian(NEE_masked))
            
            nee_bin_members.append(np.array(nee_bin)) 
    
    NEE_2010 =  NEE_RECCAP.sel(time= slice('2009-01-01', '2011-12-31')).mean(dim='time').median(dim = "ensemble_member") * 1e+15
    NEE_2010 = NEE_2010 - lateral_fluxes_sink_2010 + lateral_fluxes_source_2010
    NEE_2010 = NEE_2010.where(forest_fraction>0)
    NEE_2010_filtered = filter_nan_gaussian_conserving(NEE_2010, length_km=500)
    NEE_2010_filtered = NEE_2010_filtered.reindex(latitude=NEE_2010_filtered.latitude[::-1])
    NEE_2010_filtered['latitude'] = forest_fraction['latitude']
    NEE_2010_filtered['longitude'] = forest_fraction['longitude']
    NEE_2010_filtered = NEE_2010_filtered.where(forest_fraction>0)
    if management_type == 'intact_forests':
        NEE_2010_filtered = NEE_2010_filtered.where(management_fraction[[management_type]].to_array().sum(dim="variable")>0.5)
    else:
        NEE_2010_filtered = NEE_2010_filtered.where(management_fraction[management_type].to_array().sum(dim="variable")>0.5)

    NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').median(dim = "ensemble_member") * 1e+15
    NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
    NEE_2020 = NEE_2020.where(forest_fraction>0)
    NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)

    NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
    NEE_2020_filtered['latitude'] = forest_fraction['latitude']
    NEE_2020_filtered['longitude'] = forest_fraction['longitude']
    NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>0)
    if management_type == 'intact_forests':
        NEE_2020_filtered = NEE_2020_filtered.where(management_fraction[[management_type]].to_array().sum(dim="variable")>0.5)
    else:
        NEE_2020_filtered = NEE_2020_filtered.where(management_fraction[management_type].to_array().sum(dim="variable")>0.5)

        
    NEE_diff_2020_2010 = (NEE_2020_filtered - NEE_2010_filtered)  #*pixel_area / 1e12

    ratio_ = ((total_stand_replaced_area.values.reshape(-1) / total_aging_area.values.reshape(-1))) /10
    nee_ = NEE_diff_2020_2010.values.reshape(-1)
    
    AgeBins = np.concatenate([np.arange(0, .6, 0.01), [1]]) /10
    nee_bin_ensemble = []
    for j in range(len(AgeBins)-1):
        Agemask = (ratio_ > AgeBins[j]) & (ratio_ <= AgeBins[j+1])
        NEE_masked = nee_[Agemask]
        nee_bin_ensemble.append(np.nanmedian(NEE_masked))
    nee_bin_ensemble = np.array(nee_bin_ensemble)
    
    #%% Plot
    width = 0.2  # Reduce the width of the bars
    
    AgeBins = np.concatenate([np.arange(0, .6, 0.01), [1]]) /10
    AgeBin_midpoints = (AgeBins[:-1] + AgeBins[1:]) / 2
    AgeBin_midpoints[-1] = 0.06
    
    #nee_bin_ensemble[nee_bin_ensemble < -0.1] = 0
    
    color_positive = "#1f77b4"  # Example color for positive slope
    color_negative = "#ff7f0e"  # Example color for negative slope
    
    positive_indices = nee_bin_ensemble >= 0
    negative_indices = nee_bin_ensemble < 0
    
    # Scatter plot for positive values
    ax[i].scatter(AgeBin_midpoints[positive_indices], nee_bin_ensemble[positive_indices], 
                     s=50, color="#d95f02")
    
    # Scatter plot for negative values
    ax[i].scatter(AgeBin_midpoints[negative_indices], nee_bin_ensemble[negative_indices], 
                     s=50, color='#7570b3')
    
    ax[i].set_ylabel('Changes in NEE [gC m$^{-2}$ year$^{-1}$]', size=14)
    #ax[i].set_xlabel(r'$\frac{Area\ stand-replaced\ forests}{Area\ gradually\ ageing\ forests}$ [year$^{-1}$]', size=14)
    
    ax[i].set_xlabel('Stand-replacement extent [year$^{-1}$]', size=14)

    ax[i].tick_params(labelsize=10)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    
    mask_nan = np.isfinite(nee_bin_ensemble)
    slope, intercept, r_value, p_value, std_err = stats.linregress(AgeBin_midpoints[mask_nan], nee_bin_ensemble[mask_nan])
    ax[i].plot(AgeBin_midpoints[mask_nan], intercept + slope * AgeBin_midpoints[mask_nan], color='black', linewidth=4)
    
    ax[i].text(.05, 0.8, f'R² = {r_value**2:.2f}\nSlope = +{slope:.2f} TgC year$^{{-2}}$\np-value = {p_value:.2f}', 
                 transform=ax[i].transAxes, fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='black'))
    
    xticks = np.arange(0, .07, 0.01)
    xticklabels = [f"{x:.2f}" for x in xticks[:-1]] + [">0.06"]
    ax[i].set_xticks(xticks)
    ax[i].set_xticklabels(xticklabels)
    
    positive_slopes = 0
    negative_slopes = 0
    significance_level = 0.05
    non_significant_slopes = 0

    slope_member = []
    for nee_bin_member in nee_bin_members:
        if np.isfinite(np.nanmean(nee_bin_member)):
            mask_nan = np.isfinite(nee_bin_ensemble)
            slope, intercept, _, p_value, _ = stats.linregress(AgeBin_midpoints[mask_nan], nee_bin_member[mask_nan])
            slope_member.append(slope)
            coefficients = np.polyfit(AgeBin_midpoints[mask_nan], nee_bin_member[mask_nan], 1)
            polynomial = np.poly1d(coefficients)
            x = np.linspace(AgeBin_midpoints[0], AgeBin_midpoints[-1], 100)
            y = polynomial(x)
            ax[i].plot(AgeBin_midpoints[mask_nan], intercept + slope * AgeBin_midpoints[mask_nan], color="darkgrey", linestyle='dashed', alpha=0.5, linewidth=1.5)
            
            if p_value < significance_level:
                if slope > 0:
                    positive_slopes += 1
                else:
                    negative_slopes += 1
            else:
                non_significant_slopes += 1

    ax[i].text(0.05, 0.15, f'(+)slope* = {positive_slopes} members', 
                 transform=ax[i].transAxes, verticalalignment='top', 
                 fontsize=12, color="#d95f02", fontweight='bold')

    ax[i].text(0.055, 0.08, f'(–)slope* = {negative_slopes} members',  # en dash
                 transform=ax[i].transAxes, verticalalignment='top', 
                 fontsize=12, color='#7570b3', fontweight='bold')


    ax[i].annotate(r'$\uparrow$C source or $\downarrow$C sink  ', xy=(.065, 2), xytext=(.065,20),
                    arrowprops=dict(facecolor='#d95f02', arrowstyle="<-", linewidth=2),
                    ha='center', va='bottom', color='#d95f02', fontweight= 'bold', fontsize=13)

    # Annotate for 'Carbon Sink' below 0
    ax[i].annotate(r'$\uparrow$C sink', xy=(.065, -2), xytext=(.065, -15),
                    arrowprops=dict(facecolor= '#7570b3', arrowstyle="<-", linewidth=2),
                    ha='center', va='bottom', color= '#7570b3', fontweight= 'bold', fontsize=13)

    ax[i].axhline(y=0, c='red', linestyle='dashed', linewidth=2)
    ax[i].set_xlim(0,.065)
    if i == 0:
        ax[i].text(0.05, 1.05, 'a', transform=ax[i].transAxes,
                    fontsize=16, fontweight='bold', va='top')
        ax[i].set_title('Fraction of unmanaged forests >0.5', fontweight= 'bold')

    else:
        ax[i].text(0.05, 1.05, 'b', transform=ax[i].transAxes,
                    fontsize=16, fontweight='bold', va='top')
        ax[i].set_title('Fraction of managed forests >0.5', fontweight= 'bold')
        
    ax[i].set_ylim(-20, 50)

    i+=1    
plt.savefig(os.path.join(plot_dir,'figS8.png'), dpi=300)
