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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
from ageUpscaling.utils.plotting import violins

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

#%% Load data
df= pd.read_csv(os.path.join(data_dir,"age_biomass_management.csv"))
managed_forest = df.loc[df['management_category'] == 1]
unmanaged_forest = df.loc[df['management_category'] == 0]
plantation_forest = df.loc[df['management_category'] == 2]

#%% Plot results
AgeBins = np.concatenate([np.arange(0, 120, 20),  np.array([200, 300])])

fig, axes = plt.subplots(2, 3, figsize=(20, 11), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)
axes = axes.flatten()

for i, region_ in enumerate(np.unique(df['region'])):
    
    ax = axes[i]

    for j in range(len(AgeBins)-1):
        age_managed = managed_forest.loc[managed_forest['region'] == region_].forest_age
        agb_managed = managed_forest.loc[managed_forest['region'] == region_].biomass * 0.5
        
        Agemask = (age_managed.values.reshape(-1) > AgeBins[j]) & (age_managed.values.reshape(-1) <= AgeBins[j+1])
        AGB_masked = agb_managed.values.reshape(-1)[Agemask]
        AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
        IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.95)) & (AGB_masked > np.quantile(AGB_masked, 0.05))
        positive_values = AGB_masked[IQ_mask]
        
        # Calculate points for positive and negative values
        pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
        
        # Plot positive values in red
        ax.scatter(pointy_pos - 0.25, pointx_pos, color='#66c2a5', alpha=0.2, marker='.')
        
        # Plot the mean as a large diamond
        ax.scatter(j - 0.25, np.nanquantile(AGB_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
        
    for j in range(len(AgeBins)-1):
        try:

            age_unmanaged = unmanaged_forest.loc[unmanaged_forest['region'] == region_].forest_age
            agb_unmanaged = unmanaged_forest.loc[unmanaged_forest['region'] == region_].biomass * 0.5
            
            Agemask = (age_unmanaged.values.reshape(-1) > AgeBins[j]) & (age_unmanaged.values.reshape(-1) <= AgeBins[j+1])
            AGB_masked = agb_unmanaged.values.reshape(-1)[Agemask]
            
            AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
            IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.95)) & (AGB_masked > np.quantile(AGB_masked, 0.05))
            positive_values = AGB_masked[IQ_mask]
            
            # Calculate points for positive and negative values
            pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
            
            # Plot positive values in red
            
            ax.scatter(pointy_pos, pointx_pos, color='#fc8d62', alpha=0.2, marker='.')
            
            # Plot the mean as a large diamond
            ax.scatter(j, np.nanquantile(AGB_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
        except:
            print('no data')
        
    for j in range(len(AgeBins)-1):
        try:
            age_plantation = plantation_forest.loc[plantation_forest['region'] == region_].forest_age
            agb_plantation = plantation_forest.loc[plantation_forest['region'] == region_].biomass * 0.5
            
            Agemask = (age_plantation.values.reshape(-1) > AgeBins[j]) & (age_plantation.values.reshape(-1) <= AgeBins[j+1])
            AGB_masked = agb_plantation.values.reshape(-1)[Agemask]
            
            AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
            IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.95)) & (AGB_masked > np.quantile(AGB_masked, 0.05))
            positive_values = AGB_masked[IQ_mask]
            
            # Calculate points for positive and negative values
            pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
            
            # Plot positive values in red
            
            ax.scatter(pointy_pos + 0.25, pointx_pos, color='#8da0cb', alpha=0.2, marker='.')
            
            # Plot the mean as a large diamond
            ax.scatter(j + 0.25, np.nanquantile(AGB_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
        except:
            print('no data')
        
    ax.scatter([], [], color='#66c2a5', marker='.', s=200, label='managed')
    ax.scatter([], [], color='#fc8d62', marker='.', s=200,label='unmanaged')
    ax.scatter([], [], color='#8da0cb', marker='.', s=200,label='plantation')
        
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
plt.savefig(os.path.join(plot_dir,'figS11.png'), dpi=300)


