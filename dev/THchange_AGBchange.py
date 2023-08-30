#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:00:48 2022

@author: sbesnard
"""
#%% Load library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.optimize import curve_fit

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
dat_ = pd.read_csv('/home/simon/Downloads/harz_forestHeightagbAnalysis.csv')
delta_AGB = (dat_['agb'].values / 10) *0.5
delta_TH = dat_['deltaHeight'].values / 10

#%% Plot scatter plot
fig, ax = plt.subplots(1,1, figsize=(6,5),  gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)

AgbBins = np.arange(-25, 30, 5)
median_values = []
num_points = []

for j in range(len(AgbBins)-1):
    AGBmask = (delta_AGB >= AgbBins[j]) & (delta_AGB < AgbBins[j+1])
    TH_masked = delta_TH[AGBmask]
    IQ_mask = (TH_masked > np.quantile(TH_masked, 0.25)) & (TH_masked < np.quantile(TH_masked, 0.75))
    median_val = np.nanquantile(TH_masked[IQ_mask], 0.5)
    median_values.append(median_val)
    num_points.append(IQ_mask.sum())  # Count the number of points

    pointx, pointy, fillx, filly = violins(TH_masked[IQ_mask], pos=j, spread=0.3, max_num_points=5000)
    # plot a lightly colored traditional violin plot behind the points
    ax.fill_between(fillx, filly[0], filly[1], alpha=0.3, color='blue')      
    # plot the points from the distribution as a scatterplot
    ax.scatter(pointx, pointy, color='darkblue', alpha=0.1, marker='.')
    ax.scatter(np.nanquantile(TH_masked[IQ_mask], 0.5),j, marker='d', s=200, color='red', alpha=0.5)

ytick_positions = np.arange(len(AgbBins)-1)
ytick_labels = [f'{AgbBins[i]} to {AgbBins[i+1]}' for i in range(len(AgbBins)-1)]
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels, rotation=0, size=14)

# Fit a quadratic curve to the median values
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

popt, _ = curve_fit(quadratic, ytick_positions, median_values)
fitted_curve_values = quadratic(ytick_positions, *popt)
ax.plot(fitted_curve_values, ytick_positions, color='green', linestyle='--',linewidth=3)

# Add the number of points as text above each median value
for i, median_val in enumerate(median_values):
    ax.text(median_val, i + 0.2, f'N={num_points[i]}', ha='center', va='bottom', color='black')

ax.set_xlabel('Tree height changes [meter year$^{-1}$]', size=14)
ax.set_ylabel('Biomass changes [MgC ha$^{-1}$ year$^{-1}$]', size=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('/home/simon/Documents/science/fig1.png', dpi=300)