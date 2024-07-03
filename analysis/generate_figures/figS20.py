#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:02:10 2024

@author: simon
"""

import pandas as pd
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


#%% Load data
df= pd.read_csv("/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/age_biomass_management.csv")
managed_forest = df.loc[df['management_category'] == 1]
unmanaged_forest = df.loc[df['management_category'] == 0]
plantation_forest = df.loc[df['management_category'] == 3]

#%% Plot results
AgeBins = np.concatenate([np.arange(0, 120, 20),  np.array([200, 300])])

fig, axes = plt.subplots(2, 3, figsize=(16, 11), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)
axes = axes.flatten()

for i, region_ in enumerate(np.unique(df['region'])):
    
    ax = axes[i]

    for j in range(len(AgeBins)-1):
        age_managed = managed_forest.loc[managed_forest['region'] == region_].forest_age
        agb_managed = managed_forest.loc[managed_forest['region'] == region_].biomass * 0.5
        
        Agemask = (age_managed.values.reshape(-1) > AgeBins[j]) & (age_managed.values.reshape(-1) <= AgeBins[j+1])
        AGB_masked = agb_managed.values.reshape(-1)[Agemask]
        AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
        IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.75)) & (AGB_masked > np.quantile(AGB_masked, 0.25))
        positive_values = AGB_masked[IQ_mask]
        
        # Calculate points for positive and negative values
        pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
        
        # Plot positive values in red
        ax.scatter(pointy_pos - 0.18, pointx_pos, color='#66c2a5', alpha=0.2, marker='.')
        
        # Plot the mean as a large diamond
        ax.scatter(j - 0.18, np.nanquantile(AGB_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
        
    for j in range(len(AgeBins)-1):
        try:

            age_unmanaged = unmanaged_forest.loc[unmanaged_forest['region'] == region_].forest_age
            agb_unmanaged = unmanaged_forest.loc[unmanaged_forest['region'] == region_].biomass * 0.5
            
            Agemask = (age_unmanaged.values.reshape(-1) > AgeBins[j]) & (age_unmanaged.values.reshape(-1) <= AgeBins[j+1])
            AGB_masked = agb_unmanaged.values.reshape(-1)[Agemask]
            
            AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
            IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.75)) & (AGB_masked > np.quantile(AGB_masked, 0.25))
            positive_values = AGB_masked[IQ_mask]
            
            # Calculate points for positive and negative values
            pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
            
            # Plot positive values in red
            
            ax.scatter(pointy_pos+ 0.18, pointx_pos, color='#fc8d62', alpha=0.2, marker='.')
            
            # Plot the mean as a large diamond
            ax.scatter(j+ 0.18, np.nanquantile(AGB_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
        except:
            print('no data')
        
    # for j in range(len(AgeBins)-1):
    #     try:
    #         age_plantation = plantation_forest.loc[plantation_forest['region'] == region_].forest_age
    #         agb_plantation = plantation_forest.loc[plantation_forest['region'] == region_].biomass * 0.5
            
    #         Agemask = (age_plantation.values.reshape(-1) > AgeBins[j]) & (age_plantation.values.reshape(-1) <= AgeBins[j+1])
    #         AGB_masked = agb_plantation.values.reshape(-1)[Agemask]
            
    #         AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
    #         IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.75)) & (AGB_masked > np.quantile(AGB_masked, 0.25))
    #         positive_values = AGB_masked[IQ_mask]
            
    #         # Calculate points for positive and negative values
    #         pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
            
    #         # Plot positive values in red
            
    #         ax.scatter(pointy_pos + 0.18, pointx_pos, color='#8da0cb', alpha=0.2, marker='.')
            
    #         # Plot the mean as a large diamond
    #         ax.scatter(j + 0.18, np.nanquantile(AGB_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
    #     except:
    #         print('no data')
        
    ax.scatter([], [], color='#66c2a5', marker='.', s=200, label='managed')
    ax.scatter([], [], color='#fc8d62', marker='.', s=200,label='unmanaged')
    #ax.scatter([], [], color='#8da0cb', marker='.', s=200,label='plantation')
        
    ax.set_xticks(np.arange(0,7))
    ax.set_xticklabels(['0-20', '21-40', '41-60', '61-80', '81-100', '101-200', '>200'], rotation=0, size=14)
    ax.set_ylabel('AGB [MgC ha$^{-1}$]', size=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.05, 1.05, chr(97 + i), transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')
    ax.tick_params(labelsize=12, rotation=90)
    ax.set_title(region_, fontweight='bold')    
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    #ax.set_ylim(0, 240)

plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS20.png', dpi=300)


