#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:37:00 2024

@author: simon
"""
#%% Load library
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl
params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
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
    # tex
    #'text.usetex': True,
    # layout
    #'constrained_layout': True
}

mpl.rcParams.update(params)

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

#%% Load forest fraction
forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction
BiomassDiffPartition_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/BiomassPartition_1deg').median(dim = 'members')
Young_stand_replaced_AGR =  BiomassDiffPartition_1deg.stand_replaced_AGR.sel(age_class= '0-20')
Intermediate_stand_replaced_AGR = BiomassDiffPartition_1deg.stand_replaced_AGR.sel(age_class= '20-80')
Mature_stand_replaced_AGR =   BiomassDiffPartition_1deg.stand_replaced_AGR.sel(age_class= '80-200')
OG_stand_replaced_AGR =   BiomassDiffPartition_1deg.stand_replaced_AGR.sel(age_class= '>200')
Young_aging_AGR =  BiomassDiffPartition_1deg.gradually_ageing_AGR.sel(age_class= '0-20') 
Intermediate_aging_AGR= BiomassDiffPartition_1deg.gradually_ageing_AGR.sel(age_class= '20-80')
Mature_aging_AGR =   BiomassDiffPartition_1deg.gradually_ageing_AGR.sel(age_class= '80-200')
OG_aging_AGR =   BiomassDiffPartition_1deg.gradually_ageing_AGR.sel(age_class= '>200')

#%% Load transcom regions
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
transcom_regions['latitude'] = forest_fraction['latitude']
transcom_regions['longitude'] = forest_fraction['longitude']

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


#%% Plot data
fig, axes = plt.subplots(3, 4, figsize=(20, 18), gridspec_kw={'wspace': 0, 'hspace': 0.1}, constrained_layout=True)
axes = axes.flatten()

for j, region_ in enumerate(transcom_mask.keys()):
    class_name = transcom_mask[region_]['name']
    class_values = transcom_mask[region_]['eco_class']


    ax = axes[j]
    

    AgePartition = {
        'Young stand-replaced': Young_stand_replaced_AGR.where(transcom_regions==class_values), 'Young gradually ageing': Young_aging_AGR.where(transcom_regions==class_values),
        'Maturing stand-replaced': Intermediate_stand_replaced_AGR.where(transcom_regions==class_values), 'Maturing gradually ageing': Intermediate_aging_AGR.where(transcom_regions==class_values),
        'Mature stand-replaced': Mature_stand_replaced_AGR.where(transcom_regions==class_values), 'Mature gradually ageing': Mature_aging_AGR.where(transcom_regions==class_values),
        'OG stand-replaced': OG_stand_replaced_AGR.where(transcom_regions==class_values), 'OG gradually ageing': OG_aging_AGR.where(transcom_regions==class_values)
    }
    
    mean_agb_AGR = {}
    i = 0
    pair_gap = 0.5  # Space within pairs
    group_gap = 1.0  # Space between pairs
    tick_positions = []
    
    for idx, (key, values) in enumerate(AgePartition.items()):
        AGB_masked = values.values.reshape(-1) *100
        AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
        IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.80)) & (AGB_masked > np.quantile(AGB_masked, 0.20))
        positive_values = AGB_masked[IQ_mask]
        
        # Set color based on category
        if 'stand-replaced' in key:
            color_ = '#fc8d62'
        else: 
            color_ = '#66c2a5'

        pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=i, spread=0.3, max_num_points=1000)
        ax.scatter(pointy_pos, pointx_pos, color=color_, alpha=0.2, marker='.')
        ax.scatter(i, np.nanquantile(positive_values, 0.5), marker='d', s=200, color='black', alpha=0.5)
        mean_agb_AGR[key] = np.nanquantile(AGB_masked, 0.5)
    
        # Record position for the tick
        tick_positions.append(i)
    
        # Increment position
        if idx % 2 == 0:  # If it's the first in a pair
            i += pair_gap
        else:
            i += group_gap  # Add larger gap after the second in a pair
    
    # Set the x-ticks and x-tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(list(AgePartition.keys()), rotation=90, size=14)
    
    # Rest of your plot settings
    ax.set_ylabel('Annual growth rate [percent]', size=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right']. set_visible(False)
    ax.tick_params(labelsize=12)
    ax.set_title(class_name, fontweight='bold')
    ax.text(0.05, 1.05, chr(97 + j), transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
    #ax.set_ylim(0, 140)
fig.delaxes(axes[-1])

plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS12.png', dpi=300)

