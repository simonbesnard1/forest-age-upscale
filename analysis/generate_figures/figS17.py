#%% Load library
import xarray as xr
import numpy as np
import scipy.stats as st
import matplotlib as mpl
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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

def filter_nan_gaussian_conserving(ds_: xr.DataArray, 
                                   length_km: float = 1000, 
                                   length_degree_longitude_equator_km: float = 112.32) -> xr.DataArray:
    """
    Apply a Gaussian filter to an xarray DataArray, preserving the total intensity 
    while considering NaN values.

    This function applies a Gaussian filter to the input DataArray `ds_`. The 
    filtering conserves the total 'intensity' (sum of the values) by redistributing
    intensity only among non-NaN pixels. The NaN values in the input DataArray remain
    NaN in the output. The Gaussian distribution weights used for intensity 
    redistribution consider only available (non-NaN) pixels. 

    The smoothing scale of the Gaussian filter is determined by `length_km`, which 
    is the physical length scale in kilometers. The sigma of the Gaussian filter is 
    calculated based on the length in degrees of longitude at the equator, given by
    `length_degree_longitude_equator_km`.

    Parameters:
    ds_ (xr.DataArray): The input DataArray to be filtered. It should contain NaN 
                        values to indicate missing data.
    length_km (float, optional): The physical length scale in kilometers for the 
                                 Gaussian filter. Default is 1000 km.
    length_degree_longitude_equator_km (float, optional): The length in degrees of 
                                                          longitude at the equator, 
                                                          used for calculating sigma 
                                                          of the Gaussian filter. 
                                                          Default is 112.32 km.

    Returns:
    xr.DataArray: A new DataArray that has been smoothed with a Gaussian filter. 
                  NaN values from the original DataArray are preserved.
    """

    sigma = length_km / length_degree_longitude_equator_km 
    
    arr = ds_.values
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr
    
    out_ = xr.DataArray(gauss, coords=ds_.coords)

    return out_


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
 
def area_weighted_sum(data, res, scalar_area = 1, scalar_mass=1e-15):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res)
    dat_area_weighted = np.nansum(data * area_grid * scalar_area * scalar_mass)
    return dat_area_weighted

def area_weighted_mean(data, res):
    area_grid = AreaGridlatlon(data["latitude"].values, data["longitude"].values,res,res).values
    dat_area_weighted = np.nansum((data * area_grid) / np.nansum(area_grid[~np.isnan(data)]))
    return dat_area_weighted 

#%% Load stand-replaced / aging agb difference
forest_fraction = forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction

#%% Calculate pixel area
EARTH_RADIUS = 6371.0
delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
width_of_longitude = EARTH_RADIUS * delta_lon
delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
height_of_latitude = EARTH_RADIUS * delta_lat
pixel_area = (width_of_longitude * height_of_latitude *
              np.cos(np.deg2rad(forest_fraction.latitude))).broadcast_like(forest_fraction) * 1000000

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
NEE_2010 =  NEE_RECCAP.sel(time= slice('2009-01-01', '2011-12-31')).mean(dim='time').median(dim = "ensemble_member") * 1e+15
NEE_2010 = NEE_2010 - lateral_fluxes_sink_2010 + lateral_fluxes_source_2010
NEE_2010 = NEE_2010.where(forest_fraction>0)
NEE_2010_filtered = filter_nan_gaussian_conserving(NEE_2010, length_km=500)
#NEE_2010_filtered = apply_blur(NEE_2010)
NEE_2010_filtered = NEE_2010_filtered.reindex(latitude=NEE_2010_filtered.latitude[::-1])
NEE_2010_filtered['latitude'] = forest_fraction['latitude']
NEE_2010_filtered['longitude'] = forest_fraction['longitude']
NEE_2010_filtered = NEE_2010_filtered.where(forest_fraction>0)

NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').median(dim = "ensemble_member") * 1e+15
NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
NEE_2020 = NEE_2020.where(forest_fraction>0)
NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)
#NEE_2020_filtered = apply_blur(NEE_2020)

NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
NEE_2020_filtered['latitude'] = forest_fraction['latitude']
NEE_2020_filtered['longitude'] = forest_fraction['longitude']
NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>0)

NEE_diff_2020_2010 = (NEE_2020_filtered - NEE_2010_filtered)  #*pixel_area / 1e12

NEE_region_changes = {}

for region_ in list(transcom_mask.keys()):
    class_values = transcom_mask[region_]['eco_class']
    class_name = transcom_mask[region_]['name']
    NEE_region_changes[class_name] = area_weighted_sum(NEE_diff_2020_2010.where(transcom_regions == class_values), 1)

        
#%% Load stand-replaced data
AgeDiff_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiff_1deg').median(dim = 'members')
total_aging_area =  AgeDiff_1deg.aging_forest_class
total_stand_replaced_area = AgeDiff_1deg.stand_replaced_class

ratio_area = {}
total_area_ageing_region ={}
total_area_stand_replaced_forests_region = {}

for region_ in list(transcom_mask.keys()):
    class_values = transcom_mask[region_]['eco_class']
    class_name = transcom_mask[region_]['name']
    total_area_ageing_region[class_name] = (total_aging_area.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    total_area_stand_replaced_forests_region[class_name] = (total_stand_replaced_area.where(transcom_regions==class_values) * pixel_area * forest_fraction).sum(dim=['latitude', 'longitude']).values / 10**7
    ratio_area[class_name] = (total_area_stand_replaced_forests_region[class_name] / total_area_ageing_region[class_name]) /10
        
#%% Plot data
fig, ax = plt.subplots(1, 1, figsize=(6, 4), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

ax.scatter(ratio_area.values(), NEE_region_changes.values())
ax.set_ylabel('Changes in NEE [gC m$^{-2}$ year$^{-1}$]', size=14)
ax.set_xlabel(r'$\frac{Area\ stand-replaced\ forests}{Area\ gradually\ ageing\ forests}$ [year$^{-2}$]', size=14)
ax.tick_params(labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(list(ratio_area.values())), np.array(list(NEE_region_changes.values())))
ax.plot(np.array(list(ratio_area.values())), intercept + slope * np.array(list(ratio_area.values())), color='black', linewidth=4)

if p_value <0.001:
    ax.text(.05, 0.8, f'R² = {r_value**2:.2f}\nSlope = +{slope:.2f} gC m$^{{-2}}$ year$^{{-2}}$\np-value<0.001', 
                 transform=ax.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='black'))
else:
    ax.text(.05, 0.8, f'R² = {r_value**2:.2f}\nSlope = +{slope:.2f} gC m$^{{-2}}$ year$^{{-1}}$\np-value>0.05', 
                 transform=ax.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='black'))
xticks = np.arange(0, .025, 0.01)
xticklabels = [f"{x:.2f}" for x in xticks[:-1]] + [">0.03"]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS17.png', dpi=300)


