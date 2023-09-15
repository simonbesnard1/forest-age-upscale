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
from matplotlib import colors
import matplotlib as mpl

def violins(data,data_z, pos=0,bw_method=None,resolution=50,spread=1,max_num_points=100):
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
        random_indices = np.random.choice(len(data), size=max_num_points, replace=False)
        data = data[random_indices]
        data_z = data_z[random_indices] 
    kde    = st.gaussian_kde(data,bw_method=bw_method)
    pointx = data
    pointz = data_z
    pointy = kde.pdf(pointx)
    pointy = pointy/(2*pointy.max())
    fillx  = np.linspace(data.min(),data.max(),resolution)
    filly  = kde.pdf(fillx)
    filly  = filly/(2*filly.max())
    pointy = pos+np.where(np.random.rand(pointx.shape[0])>0.5,-1,1)*np.random.rand(pointx.shape[0])*pointy*spread
    filly  = (pos-filly*spread,pos+filly*spread)
    return(pointx,pointy,pointz, fillx,filly)


#%% Load data
dat_ = pd.read_csv('/home/simon/Downloads/pretty_data_for_simon.csv')
delta_AGB = (dat_['agbTrue'].values / 10) *0.5
delta_TH = dat_['deltaHeight'].values / 10
dat_TH = dat_['H2010'].values
count_pixel = dat_['count']

#%% Plot scatter plot - Figure 1
fig, ax = plt.subplots(1,2, figsize=(12,6),  gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)

AgbBins = np.arange(-25, 30, 5)
median_values = []
num_points = []

for j in range(len(AgbBins)-1):
    AGBmask = (delta_AGB >= AgbBins[j]) & (delta_AGB < AgbBins[j+1])
    TH_masked = delta_TH[AGBmask]
    TH2010_masked = dat_TH[AGBmask]
    IQ_mask = (TH_masked > np.quantile(TH_masked, 0.25)) & (TH_masked < np.quantile(TH_masked, 0.75))
    median_val = np.nanquantile(TH_masked[IQ_mask], 0.5)
    median_values.append(median_val)
    num_points.append(IQ_mask.sum())  # Count the number of points

    pointx, pointy,_, fillx, filly = violins(TH_masked[IQ_mask], TH2010_masked[IQ_mask], pos=j, spread=0.3, max_num_points=2000)
    # plot a lightly colored traditional violin plot behind the points
    #ax[0].fill_between(fillx, filly[1], filly[0], alpha=0.3, color='blue')      
    # plot the points from the distribution as a scatterplot
    ax[0].scatter(pointy, pointx, color='darkblue', alpha=0.1, marker='.')
    ax[0].scatter(j, np.nanquantile(TH_masked[IQ_mask], 0.5), marker='d', s=200, color='red', alpha=0.5)

ytick_positions = np.arange(len(AgbBins)-1)
ytick_labels = [f'{AgbBins[i]} to {AgbBins[i+1]}' for i in range(len(AgbBins)-1)]
ax[0].set_xticks(ytick_positions)
ax[0].set_xticklabels(ytick_labels, rotation=90, size=14)

# Fit a quadratic curve to the median values
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

popt, _ = curve_fit(quadratic, ytick_positions, median_values, )
fitted_curve_values = quadratic(ytick_positions, *popt)
ax[0].plot(ytick_positions, median_values, color='green', linestyle='--',linewidth=3)

# Add the number of points as text above each median value
# for i, median_val in enumerate(median_values):
#     ax[0].text(i + 0.2, median_val, f'N={num_points[i]}', ha='center', va='bottom', color='black')

ax[0].set_ylabel('Canopy height changes [meter year$^{-1}$]', size=14)
ax[0].set_xlabel('Biomass changes [MgC ha$^{-1}$ year$^{-1}$]', size=14)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

AgbBins = np.arange(-1.5, 1.5, .25)
median_values = []
num_points = []
# for j in range(len(AgbBins)-1):
#     AGBmask = (delta_TH >= AgbBins[j]) & (delta_TH < AgbBins[j+1])
#     TH_masked = delta_AGB[AGBmask]
#     TH2010_masked = dat_TH[AGBmask]
    
#     IQ_mask = (TH_masked > np.quantile(TH_masked, 0.25)) & (TH_masked < np.quantile(TH_masked, 0.75))
#     median_val = np.nanquantile(TH_masked[IQ_mask], 0.5)
#     median_values.append(median_val)
#     num_points.append(IQ_mask.sum())  # Count the number of points

#     pointx, pointy, pointz, fillx, filly = violins(TH_masked[IQ_mask], TH2010_masked[IQ_mask], pos=j, spread=0.3, max_num_points=2000)
#     # plot a lightly colored traditional violin plot behind the points
#     #ax[1].fill_between(fillx, filly[0], filly[1], alpha=0.3, color='blue')      
#     # plot the points from the distribution as a scatterplot
hexbin = ax[1].hexbin(delta_TH, delta_AGB, C=np.round(dat_TH, -1), cmap="afmhot_r", gridsize=80, mincnt=15)
cbar = plt.colorbar(hexbin, ax=ax[1])
cbar.set_label('Canopy height [meters]', size=14)

#x[1].scatter(j, np.nanquantile(TH_masked[IQ_mask], 0.5), marker='d', s=200, color='red', alpha=0.5)

#ytick_positions = np.arange(len(AgbBins)-1)
#ytick_labels = [f'{AgbBins[i]} to {AgbBins[i+1]}' for i in range(len(AgbBins)-1)]
#ax[1].set_xticks(ytick_positions)
#ax[1].set_xticklabels(ytick_labels, rotation=90, size=14)

# # Fit a quadratic curve to the median values
# def quadratic(x, a, b, c):
#     return a * x**2 + b * x + c

# popt, _ = curve_fit(quadratic, ytick_positions, median_values, )
# fitted_curve_values = quadratic(ytick_positions, *popt)
# ax[1].plot(ytick_positions, median_values, color='green', linestyle='--',linewidth=3)

# Add the number of points as text above each median value
# for i, median_val in enumerate(median_values):
#     ax[1].text(i + 0.2, median_val, f'N={num_points[i]}', ha='center', va='bottom', color='black')

ax[1].set_xlabel('Canopy height changes [meter year$^{-1}$]', size=14)
ax[1].set_ylabel('Biomass changes [MgC ha$^{-1}$ year$^{-1}$]', size=14)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

plt.savefig('/home/simon/Documents/science/fig1.png', dpi=300)


#%% Figure 2
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def heat_map(x_dat, y_dat, c, x_axis, y_axis):
    data = np.zeros((len(x_axis)-1, len(y_axis)-1), dtype=np.float64)
    for y in range(len(y_axis)-1):
        for x in range(len(x_axis)-1):
            with np.errstate(invalid='ignore'):
                mask = c[np.where((x_dat >= x_axis[x]) & (x_dat <=x_axis[x + 1]) &
                                  (y_dat >= y_axis[y]) & (y_dat <= y_axis[y + 1]))]
                if np.sum(~np.isnan(mask))>=10: # mask bins having less than 10 grid cells
                    data[y, x] = np.nanmedian(mask)
    return(data)

def heatmap(data, ytick, xtick, ticks, label, xlabel, ylabel, ax=None, cbar_kw={}, **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im1 = ax.pcolor(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, orientation = 'vertical', **cbar_kw)
    cbar.set_ticklabels(label)
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label('Canopy height [meters]', fontsize=10, rotation=270, labelpad=12)

    # We want to show all ticks and x y labels
    locator = mpl.ticker.MaxNLocator(nbins=4) # with 3 bins you will have 4 ticks
    ax.set_xticks(np.arange(data.shape[1] + 1))
    ax.set_yticks(np.arange(data.shape[0] + 1))
    ax.set_xticklabels(xtick, size=12)
    ax.set_yticklabels(ytick, size=12)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.set_xlabel(xlabel, size=12)
    ax.set_ylabel(ylabel, size=12)    
    
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    return im1, cbar

#%% Plot matrix
fig, ax = plt.subplots(1,1, figsize=(6,5.5), constrained_layout=True)
x_data =  delta_TH
y_data = delta_AGB
xtick =  np.arange(-3, 3, 0.25)
ytick = np.arange(-30, 30, 2.5)
x_data[x_data < xtick[0]] = xtick[0]
x_data[x_data > xtick[-1]] = xtick[-1]
y_data[y_data < ytick[0]] = ytick[0]
y_data[y_data > ytick[-1]] = ytick[-1]
heatmap_data = heat_map(x_data, y_data, dat_TH, xtick, ytick)
xlabel = 'Canopy height changes [meter year$^{-1}$]'
ylabel = 'Biomass changes [MgC ha$^{-1}$ year$^{-1}$]'

heatmap_data[heatmap_data<np.nanpercentile(heatmap_data,2)]=np.nanpercentile(heatmap_data,2)
heatmap_data[heatmap_data>np.nanpercentile(heatmap_data,98)]=np.nanpercentile(heatmap_data,98)
heatmap_data[heatmap_data==0] = np.nan
min_ = 0
max_ = 30
norm = colors.Normalize(vmin=min_, vmax=max_)
ticks = 0, 10, 20, 30
label = np.round(ticks, 2)
cmap = truncate_colormap(plt.get_cmap('afmhot_r'), 0.5, 1)
heatmap(heatmap_data, ytick, xtick, ticks, label, xlabel, ylabel, ax=ax, norm=norm, cmap=cmap, cbar_kw=dict(shrink=0.8))       
plt.savefig('/home/simon/Documents/science/fig2.png', dpi=300)


#%% Figure 3

# Define the class boundaries
class_boundaries = [0, 5, 10, 15, 20, 25]

# Create an array to store the class labels for each element in dat_TH
class_labels = np.digitize(dat_TH, class_boundaries)

# Create dictionaries to store frequencies for each category within each class
categories = {
    'biomass gain-height gain': [],
    'biomass loss-height gain': [],
    'biomass gain-height loss': [],
    'biomass loss-height loss': []
}

# Count frequencies for each category within each class
for i, label in enumerate(class_labels):
    if delta_AGB[i] > 0 and delta_TH[i] > 0:
        for pixel_ in np.arange(count_pixel[i]):
            categories['biomass gain-height gain'].append(label)
    elif delta_AGB[i] < 0 and delta_TH[i] > 0:
        for pixel_ in np.arange(count_pixel[i]):
            categories['biomass loss-height gain'].append(label)
    elif delta_AGB[i] > 0 and delta_TH[i] < 0:
        for pixel_ in np.arange(count_pixel[i]):
            categories['biomass gain-height loss'].append(label)
    elif delta_AGB[i] < 0 and delta_TH[i] < 0:
        for pixel_ in np.arange(count_pixel[i]):
            categories['biomass loss-height loss'].append(label)

# Create a single plot with dots for each class name and category frequency
class_names = [f"{class_boundaries[i]}-{class_boundaries[i+1]}" for i in range(len(class_boundaries) - 1)]

fig, ax = plt.subplots(1,1, figsize=(8,6),  gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)

# Plot dots for each category
for i, category in enumerate(categories.keys()):
    category_counts = [categories[category].count(i) for i in range(1, len(class_boundaries))]
    if category == 'biomass gain-height gain':
        color_ = "#CCFEA8" 
    if category == 'biomass loss-height gain':
        color_ = "#DD78B4" 
    if category == 'biomass gain-height loss':
        color_ =  "#3D6AFF"
    if category == 'biomass loss-height loss':
        color_ = 'grey'
        
    ax.plot(class_names, category_counts, label=category, color= color_, marker = '.', markersize = 20, linestyle = '--')
ax.set_xlabel('Canopy height [meter]', size=14)
ax.set_ylabel('Frequency [-]', size=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(frameon=False, fontsize=12)
plt.grid(True)
plt.savefig('/home/simon/Documents/science/fig3.png', dpi=300)


#%% Figure 4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# Load your data
dat_ = pd.read_csv('/home/simon/Downloads/pretty_data_for_simon.csv')
Y = (dat_['agbTrue'].values / 10) * 0.5
X = dat_['deltaHeight'].values / 10
Z = dat_['H2010'].values

# Create a Delaunay triangulation
triangulation = Delaunay(np.column_stack((X, Y)))

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot using plot_trisurf
surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', triangles=triangulation.simplices)

# Add labels and a color bar
ax.set_xlabel('Canopy height changes [meter year$^{-1}$]')
ax.set_ylabel('Biomass changes [MgC ha$^{-1}$ year$^{-1}$]')
ax.set_zlabel('Canopy height [meter]')
fig.colorbar(surf)

# Show the plot
plt.show()
plt.savefig('/home/simon/Documents/science/fig4.png', dpi=300)

