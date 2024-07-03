#%% Load library
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import pandas as pd
import matplotlib.colors as colors

def rolling_window(data, window_size=10):
    latitudes = data.latitude.values
    longitudes = data.longitude.values

    mean_values = []

    for lat in range(int(latitudes.min()), int(latitudes.max()) - window_size + 1):
        for lon in range(int(longitudes.min()), int(longitudes.max()) - window_size + 1):
            # Select data in the current window
            window_data = data.sel(latitude=slice(lat + window_size, lat),
                                   longitude=slice(lon, lon + window_size))
            # Calculate area-weighted mean for this window
            window_mean = window_data.mean(dim=('latitude', 'longitude')).values
            mean_values.append([window_mean])
    
    return np.concatenate(mean_values)

#%% Define functions
def AreaGridlatlon(lats,lons,res_lat,res_lon):
     ER          = 6378160 #Earth radius (m)
     cols        = lons.shape[0]
     londel      = np.abs(res_lon)
     lats1       = lats - res_lat/2.
     lats2       = lats + res_lat/2.
     areavec     = (np.pi/180)*ER**2 * np.abs(np.sin(lats1 * 
                                                     np.pi/180)-np.sin(lats2 * np.pi/180))*londel
     area_grid   = xr.DataArray(np.matmul(areavec[:,np.newaxis],np.ones([1, cols])), 
                               dims=['latitude', 'longitude'],
                               coords={'latitude': lats,
                                       'longitude': lons})
     return(area_grid)
 
def area_weighed_sum(data, res, scalar_area = 0.0001, scalar_mass=1e-15):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res)
    dat_area_weighted = np.nansum(data * area_grid * scalar_area * scalar_mass)
    return dat_area_weighted

def area_weighed_mean(data, res):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res).values
    dat_area_weighted = np.nansum((data * area_grid) / np.nansum(area_grid[~np.isnan(data)]))
    return dat_area_weighted 

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

#%% Compute area weighted age
age_2010 = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/ForestAge_fraction_1deg').sel(time = '2010-01-01').forest_age
age_2020 = xr.open_zarr('/home/simon/gfz_hpc/projects/forest-age-upscale/output/upscaling/Age_upscale_100m/XGBoost/version-1.1/ForestAge_fraction_1deg').sel(time = '2020-01-01').forest_age

# Define the midpoint for each age class
age_midpoints = {
    '0-20': 10,
    '20-40': 30,
    '40-60': 50,
    '60-80':70,
    '80-100':90,
    '100-120':110,
    '120-140':130,
    '140-160':150,
    '160-180':170,
    '180-200':190,
    '>200':200
}

# Initialize an empty DataArray for the weighted sum
weighted_mean_age_2010 = xr.zeros_like(age_2010.isel(age_class=0))
weighted_mean_age_2020 = xr.zeros_like(age_2020.isel(age_class=0))

# Iterate over the age classes and calculate the weighted sum
for age_class, midpoint in age_midpoints.items():
    weighted_mean_age_2010 += age_2010.sel(age_class = age_class) * midpoint
    weighted_mean_age_2020 += age_2020.sel(age_class = age_class) * midpoint
    
# If the total_fraction is not all ones, then you need to normalize
# by dividing the weighted sum by the total fraction
weighted_mean_age_2010 = weighted_mean_age_2010.where(weighted_mean_age_2010>0)
weighted_mean_age_2020 = weighted_mean_age_2020.where(weighted_mean_age_2020>0)
weighted_mean_age_2010_2020 =  xr.concat([weighted_mean_age_2010, weighted_mean_age_2020], dim= 'time').mean(dim = 'time')
#age_windows = rolling_window(weighted_mean_age_2010_2020.astype('int16'), window_size=10)
#weighted_mean_age_2010_2020_5deg = weighted_mean_age_2010_2020.coarsen(latitude =2, longitude=2).mean()

#%% Compute mean annual NEE
# NEE_2010_2020 = []
# for year_ in np.arange(2009, 2022):
#     NEE_annual = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/inversion_jena/s99oc_v2022_daily/NEE.daily.360.180.{year_}.nc'.format(year_ = year_))
#     NEE_2010_2020.append(NEE_annual.NEE.sum(dim = 'time').to_dataset(name = str(year_)))

# NEE_2010_2020 = xr.merge(NEE_2010_2020).to_array().median(dim = 'variable')
land_fraction = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/inversion_jena/s99oc_v2022_daily/NEE.daily.360.180.2020.nc').LF
# NEE_2010_2020 = NEE_2010_2020.where(land_fraction>0)
#NEE_2010_2020 = NEE_2010_2020.where(np.isfinite(weighted_mean_age_2010_2020))
NEE_2010_2020 =  xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/GCP/GCP2022_inversions_1x1_version1_2_20230428.nc').land_flux_fossil_cement_and_river_adjusted.sel(time= slice('2010-01-01', '2020-12-31')).median(dim = "ensemble_member").mean(dim='time') * 1e+15
NEE_2010_2020 = NEE_2010_2020.reindex(latitude=NEE_2010_2020.latitude[::-1])
NEE_2010_2020['longitude'] = weighted_mean_age_2010_2020['longitude']
NEE_2010_2020['latitude'] = weighted_mean_age_2010_2020['latitude']
NEE_2010_2020 = NEE_2010_2020.where(land_fraction>0)
#NEE_windows = rolling_window(weighted_mean_age_2010_2020.astype('int16'), window_size=10)

#%% Create heatmap matrix
class_ranges = {
    'Young forests \n (0-20 years)': ['0-20'],
    'Maturing forests \n (21-80 years)': ['20-40','40-60', '60-80'],
    'Mature forests \n (81-200 years) ': ['80-100', '100-120', '120-140', '140-160', '160-180', '180-200'],
    'Old-Growth forests \n (>200 years)':['>200']
    }
NEE_bins = [-np.inf, -150, -50,  0,  50, 150, np.inf]
NEE_labels = ['≤ -150', '-150 to -50', '-50 to 0', '0 to 50', '50 to 150', '≥ 150']
NEE_categories = np.digitize(NEE_2010_2020, NEE_bins) - 1

# Aggregating Data by Age Class
age_class_totals = {class_name: age_2010.sel(age_class = class_ranges[class_name]).sum(dim='age_class') for class_name in class_ranges}

# Initializing the Heatmap Matrix
heatmap_matrix = pd.DataFrame(0, index=NEE_labels, columns=list(class_ranges.keys()))

# Count Fractions in Carbon Gain Categories
for class_name, total_fraction in age_class_totals.items():
    for i, label in enumerate(NEE_labels):
        mask = NEE_categories == i
        heatmap_matrix.loc[label, class_name ] = total_fraction.where(mask).sum().values / total_fraction.sum().values

#%% Plot scatter plot
fig, ax = plt.subplots(1,2, figsize=(10,5),  gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)
AgeBins = np.concatenate([np.arange(0, 120, 20),  np.array([150, 200])])

for j in range(len(AgeBins)-1):
    Agemask = (weighted_mean_age_2010_2020.values.reshape(-1) > AgeBins[j]) & (weighted_mean_age_2010_2020.values.reshape(-1) <= AgeBins[j+1])
    NEE_masked = NEE_2010_2020.values.reshape(-1)[Agemask]
    NEE_masked = NEE_masked[np.isfinite(NEE_masked)]
    IQ_mask = (NEE_masked < np.quantile(NEE_masked, 0.80)) & (NEE_masked > np.quantile(NEE_masked, 0.20))
    positive_values = NEE_masked[IQ_mask][NEE_masked[IQ_mask] >= 0]
    negative_values = NEE_masked[IQ_mask][NEE_masked[IQ_mask] < 0]

    # Calculate points for positive and negative values
    pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
    pointx_neg, pointy_neg, _, _ = violins(negative_values, pos=j, spread=0.3, max_num_points=1000)

    # Plot positive values in red
    ax[0].scatter(pointy_pos - 0.16, pointx_pos, color='#d95f02', alpha=0.2, marker='.')
    
    # Plot negative values in blue
    ax[0].scatter(pointy_neg - 0.16, pointx_neg, color='#7570b3', alpha=0.2, marker='.')

    # Plot the mean as a large diamond
    ax[0].scatter(j - 0.16, np.nanquantile(NEE_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
    
    
ax[0].set_xticks(np.arange(0,7))
ax[0].set_xticklabels(['0-20', '21-40', '41-60', '61-80', '81-100', '101-150', '>200'], rotation=0, size=14)
#ax1.set_xlabel('Age class', size=12)   
ax[0].set_ylabel('NEE [gC m$^{-2}$ year$^{-1}$]', size=14)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].text(0.05, 1.1, 'a', transform=ax[0].transAxes,
            fontsize=16, fontweight='bold', va='top')
ax[0].tick_params(labelsize=12, rotation=90)
ax[0].axhline(y=0, c='red', linestyle='dashed', linewidth=2)

# Annotate for 'Carbon Source' above 0
ax[0].annotate('Carbon source', xy=(6.2, 9), xytext=(6.2, 50),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=2),
                  ha='center', va='bottom', color='#d95f02', fontweight= 'bold')

# Annotate for 'Carbon Sink' below 0
ax[0].annotate('Carbon sink', xy=(6.2, -9), xytext=(6.2, -60),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=2),
                  ha='center', va='bottom', color='#7570b3', fontweight= 'bold')

cmap = sns.light_palette("orange", 5)
#min_val, max_val = heatmap_matrix.min().min(), heatmap_matrix.max().max()
#boundaries = np.linspace(min_val, max_val, 6)  # 5 bins will have 6 boundaries
boundaries = np.arange(0, 0.6, 0.1)
norm = colors.BoundaryNorm(boundaries, len(cmap), clip=True)
sns_heatmap = sns.heatmap(heatmap_matrix.iloc[::-1], cmap=cmap, ax=ax[1], norm=norm)
cbar = ax[1].collections[0].colorbar
cbar.set_label('Age class fraction', size=12)
ax[1].set_ylabel('NEE [gC m$^{-2}$ year$^{-1}$]', size=14)
ax[1].text(0.05, 1.1, 'b', transform=ax[1].transAxes,
            fontsize=16, fontweight='bold', va='top')
ax[1].axhline(y=3, c='red', linestyle='dashed', linewidth=2)
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS7.png', dpi=300)


