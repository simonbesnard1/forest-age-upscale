#%% Load library
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.stats as st
import matplotlib as mpl
from scipy import ndimage

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

def blur_func(num, sigma):
    """Kernel density function
    Currently Gaussian, but this can change to need.
    """
    assert num % 2 == 1, 'Only with odd numbers!'
    kernel = np.zeros(num)
    mu = num // 2
    for x in range(num):
        # Could vectorize
        density = (1/(sigma * (2 * np.pi) ** 0.5) * np.exp(-1/2 * ((x - mu) ** 2) / sigma ** 2))
        kernel[x] = density
    return kernel

def get_kernel(sigma_x, sigma_y, kernel_size):
    """Function to create a 2d kernel based on the creation of 2 1d kernels. """
    kernel_x = blur_func(kernel_size, sigma_x)
    kernel_y = blur_func(kernel_size, sigma_y)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum() # Normalize to 1
    return kernel

def apply_blur(NEE_2010, kernel_size=5, sigmax=2):
    HALF_KERNEL = int(np.ceil(kernel_size / 2))
    
    # Calculate the number of pixels per latitude bin
    global_lats = NEE_2010.latitude.values
    sigma_per_lat = 1/np.cos(global_lats / 90) * sigmax

    newmap = NEE_2010.values.copy()
    for i in range(NEE_2010.shape[0]):
        for j in range(NEE_2010.shape[1]):
            
            sigmay = sigma_per_lat[i]
            kernel = get_kernel(sigmax, sigmay, kernel_size)
            starti = max(0, i - (HALF_KERNEL - 1))
            endi = min(newmap.shape[0], i + HALF_KERNEL)
            
            startj = max(0, j - (HALF_KERNEL - 1))
            endj = min(newmap.shape[1], j + HALF_KERNEL)

            if i - HALF_KERNEL < 0:
                kernel = kernel[-(i - HALF_KERNEL + 1):, :]
            elif i + HALF_KERNEL > newmap.shape[0]:
                kernel = kernel[:newmap.shape[0] - (i + HALF_KERNEL), :]
                
            if j - HALF_KERNEL < 0:
                kernel = kernel[:, -(j - HALF_KERNEL + 1):]
            elif j + HALF_KERNEL > newmap.shape[1]:
                kernel = kernel[:, :newmap.shape[1] - (j + HALF_KERNEL)]

            a = newmap[starti: endi,
                    startj: endj]

            assert a.shape == kernel.shape

            newmap[i, j] = np.average(a, weights=kernel)
    
    return xr.DataArray(newmap, coords=NEE_2010.coords)

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

#%% Compute area weighted age
age_2010 = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeClass_1deg').sel(time = '2010-01-01').forest_age.median(dim = 'members')
age_2020 = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeClass_1deg').sel(time = '2020-01-01').forest_age.median(dim = 'members')

weighted_mean_age_2010 = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestAge_1deg').sel(time = '2010-01-01').forest_age.median(dim = 'members').astype("int16")
weighted_mean_age_2010 = weighted_mean_age_2010.where(weighted_mean_age_2010>0)
weighted_mean_age_2020 = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestAge_1deg').sel(time = '2020-01-01').forest_age.median(dim = 'members').astype("int16")
weighted_mean_age_2020 = weighted_mean_age_2020.where(weighted_mean_age_2020>0)
#age_windows_2010 = rolling_window(weighted_mean_age_2010, window_size=5)
#age_windows_2020 = rolling_window(weighted_mean_age_2020, window_size=5)

#%% Load stand-replaced / aging agb difference
forest_fraction = forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction
BiomassDiffPartition_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/BiomassPartition_1deg').median(dim = 'members')
Young_stand_replaced =  (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '0-20') * 0.47)
Intermediate_stand_replaced = (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '20-80') * 0.47)
Mature_stand_replaced =   (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '80-200') * 0.47)
OG_stand_replaced =   (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '>200')* 0.47)
Young_aging =  (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '0-20') * 0.47) 
Intermediate_aging= (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '20-80') * 0.47)
Mature_aging =   (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '80-200') * 0.47)
OG_aging =   (BiomassDiffPartition_1deg.gradually_ageing.sel(age_class= '>200')* 0.47)

#%% Load stand-replaced / aging agb difference
BiomassDiffPartition_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/BiomassDiffPartition_1deg').median(dim = 'members')
Young_stand_replaced_AGB =  (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '0-20') * 0.47 *-1 *100)/ 10
Intermediate_stand_replaced_AGB = (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '20-80') * 0.47 *-1 *100)/ 10
Mature_stand_replaced_AGB =   (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '80-200') * 0.47 *-1 *100)/ 10
OG_stand_replaced_AGB =   (BiomassDiffPartition_1deg.stand_replaced.sel(age_class= '>200') * 0.47 *-1 *100)/ 10

#%% Load stand-replace age class data
AgeDiffPartition_fraction_1deg =  xr.open_zarr("/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiffPartition_1deg").median(dim='members')

Young_stand_replaced_class =  AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '0-20')
Young_stand_replaced_class = Young_stand_replaced_class.where(Young_stand_replaced_class >0)

Intermediate_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '20-80')
Intermediate_stand_replaced_class = Intermediate_stand_replaced_class.where(Intermediate_stand_replaced_class >0)

Mature_stand_replaced_class =  AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '80-200')
Mature_stand_replaced_class = Mature_stand_replaced_class.where(Mature_stand_replaced_class >0)

OG_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '>200')
OG_stand_replaced_class = OG_stand_replaced_class.where(OG_stand_replaced_class >0)

Young_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '0-20')
Young_aging_class = Young_aging_class.where(Young_aging_class >0)

Intermediate_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '20-80')
Intermediate_aging_class = Intermediate_aging_class.where(Intermediate_aging_class >0)

Mature_aging_class =  AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '80-200')
Mature_aging_class = Mature_aging_class.where(Mature_aging_class >0)

OG_aging_class = AgeDiffPartition_fraction_1deg.aging_forest_class_partition.sel(age_class = '>200')
OG_aging_class = OG_aging_class.where(OG_aging_class >0)

#%% Calculate pixel area
EARTH_RADIUS = 6371.0
delta_lon = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
width_of_longitude = EARTH_RADIUS * delta_lon
delta_lat = np.deg2rad(1)  # Assuming a grid spacing of 1 degree
height_of_latitude = EARTH_RADIUS * delta_lat
pixel_area = (width_of_longitude * height_of_latitude *
              np.cos(np.deg2rad(Intermediate_stand_replaced.latitude))).broadcast_like(Intermediate_stand_replaced) * 1000000

#%% Calculate total per area
AgeDiff_1deg =  xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/AgeDiff_1deg').median(dim = 'members')
total_aging_area =  AgeDiff_1deg.aging_forest_class
total_stand_replaced_area = AgeDiff_1deg.stand_replaced_class

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
model_name = xr.open_dataset('/home/simon/Documents/science/research_paper/global_age_Cdyn/data/NEE_inversion/GCP/GCP2023_inversions_1x1_version1_1_20240124.nc').ensemble_member_name.values
model_names = [''.join(row).strip() for row in model_name]
nee_bin_members = []
for member_ in NEE_RECCAP.ensemble_member:
    NEE_2010 =  NEE_RECCAP.sel(time= slice('2009-01-01', '2011-12-31')).isel(ensemble_member= member_).mean(dim='time') * 1e+15
    if not np.isnan(NEE_2010.values).all():
        NEE_2010 = NEE_2010 - lateral_fluxes_sink_2010 + lateral_fluxes_source_2010
        NEE_2010 = NEE_2010.where(forest_fraction>0)
        NEE_2010_filtered = filter_nan_gaussian_conserving(NEE_2010, length_km=500)
        #NEE_2010_filtered = apply_blur(NEE_2010.where(np.isfinite(NEE_2010), 0))
        NEE_2010_filtered = NEE_2010_filtered.reindex(latitude=NEE_2010_filtered.latitude[::-1])
        NEE_2010_filtered['latitude'] = Young_stand_replaced_class['latitude']
        NEE_2010_filtered['longitude'] = Young_stand_replaced_class['longitude']
        NEE_2010_filtered = NEE_2010_filtered.where(forest_fraction>0)
        
        NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).isel(ensemble_member= member_).mean(dim='time') * 1e+15
        NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
        NEE_2020 = NEE_2020.where(forest_fraction>0)
        NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)
        #NEE_2020_filtered = apply_blur(NEE_2020.where(np.isfinite(NEE_2020), 0))
        NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
        NEE_2020_filtered['latitude'] = Young_stand_replaced_class['latitude']
        NEE_2020_filtered['longitude'] = Young_stand_replaced_class['longitude']
        NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>0)
        
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
#NEE_2010_filtered = apply_blur(NEE_2010.where(np.isfinite(NEE_2010), 0))
NEE_2010_filtered = NEE_2010_filtered.reindex(latitude=NEE_2010_filtered.latitude[::-1])
NEE_2010_filtered['latitude'] = Young_stand_replaced_class['latitude']
NEE_2010_filtered['longitude'] = Young_stand_replaced_class['longitude']
NEE_2010_filtered = NEE_2010_filtered.where(forest_fraction>0)

NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).mean(dim='time').median(dim = "ensemble_member") * 1e+15
NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
NEE_2020 = NEE_2020.where(forest_fraction>0)
NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)
#NEE_2020_filtered = apply_blur(NEE_2020.where(np.isfinite(NEE_2020), 0))

NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
NEE_2020_filtered['latitude'] = Young_stand_replaced_class['latitude']
NEE_2020_filtered['longitude'] = Young_stand_replaced_class['longitude']
NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>0)

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
fig, ax = plt.subplots(2, 2, figsize=(12, 11), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

AgeBins = np.concatenate([np.arange(0, .6, 0.01), [1]]) /10
AgeBin_midpoints = (AgeBins[:-1] + AgeBins[1:]) / 2
AgeBin_midpoints[-1] = 0.06

#nee_bin_ensemble[nee_bin_ensemble < -0.05] = 0

color_positive = "#1f77b4"  # Example color for positive slope
color_negative = "#ff7f0e"  # Example color for negative slope

positive_indices = nee_bin_ensemble >= 0
negative_indices = nee_bin_ensemble < 0

# Scatter plot for positive values
ax[0, 0].scatter(AgeBin_midpoints[positive_indices], nee_bin_ensemble[positive_indices], 
                 s=50, color="#d95f02",)

# Scatter plot for negative values
ax[0, 0].scatter(AgeBin_midpoints[negative_indices], nee_bin_ensemble[negative_indices], 
                 s=50, color='#7570b3')

ax[0, 0].set_ylabel('Changes in NEE [gC m$^{-2}$ year$^{-1}$]', size=14)
#ax[0, 0].set_xlabel(r'$\frac{Area\ stand-replaced\ forests}{Area\ gradually\ ageing\ forests}$ [year$^{-1}$]', size=14)
ax[0, 0].set_xlabel('Stand-replacement extent [year$^{-1}$]', size=14)
ax[0, 0].tick_params(labelsize=10)
ax[0, 0].spines['top'].set_visible(False)
ax[0, 0].spines['right'].set_visible(False)

slope, intercept, r_value, p_value, std_err = stats.linregress(AgeBin_midpoints, nee_bin_ensemble)
ax[0, 0].plot(AgeBin_midpoints, intercept + slope * AgeBin_midpoints, color='black', linewidth=4)

if p_value <0.001:
    ax[0, 0].text(.05, 0.8, f'R² = {r_value**2:.2f}\nSlope = +{slope:.2f} gC m$^{{-2}}$ year$^{{-2}}$\np-value<0.001', 
                 transform=ax[0, 0].transAxes, fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='black'))
else:
    ax[0, 0].text(.05, 0.8, f'R² = {r_value**2:.2f}\nSlope = +{slope:.2f} gC m$^{{-2}}$ year$^{{-1}}$\np-value>0.05', 
                 transform=ax[0, 0].transAxes, fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='black'))
xticks = np.arange(0, .065, 0.01)
xticklabels = [f"{x:.2f}" for x in xticks[:-1]] + [">0.06"]
ax[0, 0].set_xticks(xticks)
ax[0, 0].set_xticklabels(xticklabels)

positive_slopes = 0
negative_slopes = 0
significance_level = 0.05
non_significant_slopes = 0

slope_member = []
for nee_bin_member in nee_bin_members:
    if np.isfinite(np.mean(nee_bin_member)):
        slope, _, _, p_value, _ = stats.linregress(AgeBin_midpoints, nee_bin_member)
        slope_member.append(slope)
        coefficients = np.polyfit(AgeBin_midpoints, nee_bin_member, 1)
        polynomial = np.poly1d(coefficients)
        x = np.linspace(AgeBin_midpoints[0], AgeBin_midpoints[-1], 100)
        y = polynomial(x)
        ax[0, 0].plot(x, y, color="darkgrey", linestyle='dashed', alpha=0.5, linewidth=1.5)
        
        if p_value < significance_level:
            if slope > 0:
                positive_slopes += 1
            else:
                negative_slopes += 1
        else:
            non_significant_slopes += 1

ax[0, 0].text(0.05, 0.15, f'(+)slope* = {positive_slopes} members', 
             transform=ax[0, 0].transAxes, verticalalignment='top', 
             fontsize=12, color="#d95f02", fontweight='bold')

ax[0, 0].text(0.055, 0.08, f'(–)slope* = {negative_slopes} members',  # en dash
             transform=ax[0, 0].transAxes, verticalalignment='top', 
             fontsize=12, color='#7570b3', fontweight='bold')


ax[0,0].annotate(r'$\uparrow$C source or $\downarrow$C sink  ', xy=(.065, 2), xytext=(.065,20),
                arrowprops=dict(facecolor='#d95f02', arrowstyle="<-", linewidth=2),
                ha='center', va='bottom', color='#d95f02', fontweight= 'bold', fontsize=13)

# Annotate for 'Carbon Sink' below 0
ax[0,0].annotate(r'$\uparrow$C sink', xy=(.065, -2), xytext=(.065, -8),
                arrowprops=dict(facecolor= '#7570b3', arrowstyle="<-", linewidth=2),
                ha='center', va='bottom', color= '#7570b3', fontweight= 'bold', fontsize=13)
ax[0,0].axhline(y=0, c='red', linestyle='dashed', linewidth=2)
ax[0,0].set_xlim(0,.065)
ax[0,0].set_ylim(-10,25)

ax[0,0].text(0.05, 1.1, 'a', transform=ax[0,0].transAxes,
            fontsize=16, fontweight='bold', va='top')

AgeBins = np.concatenate([np.arange(0, 120, 20),  np.array([200, 300])])

for j in range(len(AgeBins)-1):
    Agemask = (weighted_mean_age_2010.values.reshape(-1) > AgeBins[j]) & (weighted_mean_age_2010.values.reshape(-1) <= AgeBins[j+1])
    #Agemask = (age_windows_2010 > AgeBins[j]) & (age_windows_2010 <= AgeBins[j+1])            
    NEE_masked = NEE_2010_filtered.values.reshape(-1)[Agemask]
    #NEE_masked = NEE_windows_2010[Agemask]
    NEE_masked = NEE_masked[np.isfinite(NEE_masked)]
    IQ_mask = (NEE_masked < np.quantile(NEE_masked, 0.75)) & (NEE_masked > np.quantile(NEE_masked, 0.25))
    positive_values = NEE_masked[IQ_mask]
    
    # Calculate points for positive and negative values
    pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
    
    # Plot positive values in red
    ax[0,1].scatter(pointy_pos - 0.18, pointx_pos, color='#1f78b4', alpha=0.2, marker='.')
    
    # Plot the mean as a large diamond
    ax[0,1].scatter(j - 0.18, np.nanquantile(NEE_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
    
for j in range(len(AgeBins)-1):
    Agemask = (weighted_mean_age_2020.values.reshape(-1) > AgeBins[j]) & (weighted_mean_age_2020.values.reshape(-1) <= AgeBins[j+1])
    #Agemask = (age_windows_2020 > AgeBins[j]) & (age_windows_2020 <= AgeBins[j+1])            
    NEE_masked = NEE_2020_filtered.values.reshape(-1)[Agemask]
    #NEE_masked = NEE_windows_2020[Agemask]
    
    NEE_masked = NEE_masked[np.isfinite(NEE_masked)]
    IQ_mask = (NEE_masked < np.quantile(NEE_masked, 0.75)) & (NEE_masked > np.quantile(NEE_masked, 0.25))
    positive_values = NEE_masked[IQ_mask]
    
    # Calculate points for positive and negative values
    pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=j, spread=0.3, max_num_points=1000)
    
    # Plot positive values in red
    
    ax[0,1].scatter(pointy_pos + 0.18, pointx_pos, color='#33a02c', alpha=0.2, marker='.')
    
    # Plot the mean as a large diamond
    ax[0,1].scatter(j + 0.18, np.nanquantile(NEE_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
    
ax[0,1].scatter([], [], color='#1f78b4', marker='.', s=200, label='circa 2010')
ax[0,1].scatter([], [], color='#33a02c', marker='.', s=200,label='circa 2020')

ax[0,1].set_xticks(np.arange(0,7))
ax[0,1].set_xticklabels(['0-20', '21-40', '41-60', '61-80', '81-100', '101-200', '>200'], rotation=0, size=14)
#ax1.set_xlabel('Age class', size=12)   
ax[0,1].set_ylabel('NEE [gC m$^{-2}$ year$^{-1}$]', size=14)
ax[0,1].spines['top'].set_visible(False)
ax[0,1].spines['right'].set_visible(False)
ax[0,1].text(0.05, 1.1, 'b', transform=ax[0,1].transAxes,
            fontsize=16, fontweight='bold', va='top')
ax[0,1].tick_params(labelsize=12, rotation=90)
ax[0,1].axhline(y=0, c='red', linestyle='dashed', linewidth=2)

# Annotate for 'Carbon Source' above 0
ax[0,1].annotate('Carbon source', xy=(6.5, 4), xytext=(6.5, 20),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=2),
                  ha='center', va='bottom', color='#d95f02', fontweight= 'bold', fontsize=14)

# Annotate for 'Carbon Sink' below 0
ax[0,1].annotate('Carbon sink', xy=(6.5, -4), xytext=(6.5, -40),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=2),
                  ha='center', va='bottom', color='#7570b3', fontweight= 'bold', fontsize=14)
ax[0,1].legend(loc="lower right", frameon=False, fontsize=12)
ax[0,1].set_ylim(-100, 15)


AgePartition = {
    'Young stand-replaced': Young_stand_replaced, 'Young gradually ageing': Young_aging, 'Young difference': Young_aging -Young_stand_replaced,
    'Maturing stand-replaced': Intermediate_stand_replaced, 'Maturing gradually ageing': Intermediate_aging, 'Maturing difference': Intermediate_aging -Intermediate_stand_replaced,
    'Mature stand-replaced': Mature_stand_replaced, 'Mature gradually ageing': Mature_aging, 'Mature difference': Mature_aging - Mature_stand_replaced,
    'OG stand-replaced': OG_stand_replaced, 'OG gradually ageing': OG_aging, 'OG difference': OG_aging - OG_stand_replaced,
}

mean_agb = {}
i = 0
pair_gap = 0.75  # Space within pairs
group_gap = 2.0  # Space between groups
tick_positions = []
age_classes = ['Young', 'Maturing', 'Mature', 'OG']
age_class_positions = {}

for age_class in age_classes:
    # Define positions for stand-replaced and gradually ageing within each age class
    for sub_class in ['stand-replaced', 'gradually ageing', 'difference']:
        key = f'{age_class} {sub_class}'
        values = AgePartition[key]

        AGB_masked = values.values.reshape(-1)
        AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
        IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.80)) & (AGB_masked > np.quantile(AGB_masked, 0.20))
        positive_values = AGB_masked[IQ_mask]
        
        # Set color based on category
        if 'stand-replaced' in key:
            color_ = '#8da0cb'
        elif 'gradually ageing' in key: 
            color_ = '#66c2a5'
        else:
            color_ = '#fc8d62'            

        pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=i, spread=0.3, max_num_points=1000)
        ax[1,0].scatter(pointy_pos, pointx_pos, color=color_, alpha=0.2, marker='.')
        ax[1,0].scatter(i, np.nanquantile(positive_values, 0.5), marker='d', s=200, color='black', alpha=0.5)
        mean_agb[key] = np.nanquantile(AGB_masked, 0.5)

        # Record position for the tick
        tick_positions.append(i)

        # Increment position
        i += pair_gap

    # Add larger gap after each age class
    i += group_gap - pair_gap  # Subtract pair_gap to correct for the last addition within the age class
    age_class_positions[age_class] = i - (group_gap - pair_gap) / 2  # Position for label

# Set the x-ticks and x-tick labels
ax[1,0].set_xticks(tick_positions)
ax[1,0].set_xticklabels(list(AgePartition.keys()), rotation=90, size=14)

# Rest of your plot settings
ax[1, 0].set_ylabel('AGC stock [MgC ha$^{-1}$]', size=14)
ax[1, 0].spines['top'].set_visible(False)
ax[1, 0].spines['right']. set_visible(False)
ax[1, 0].text(0.05, 1.1, 'c', transform=ax[1, 0].transAxes, fontsize=16, fontweight='bold', va='top')
ax[1, 0].tick_params(labelsize=12)
ax[1, 0].set_ylim(0, 140)

AgePartition = {
    'Young stand-replaced': Young_stand_replaced_AGB, 
    'Maturing stand-replaced': Intermediate_stand_replaced_AGB, 
    'Mature stand-replaced': Mature_stand_replaced_AGB, 
    'OG stand-replaced': OG_stand_replaced_AGB}

mean_agb = {}

i=0
for j in list(AgePartition.keys()):
    
    AGB_masked = AgePartition[j].values.reshape(-1)
    AGB_masked = AGB_masked[np.isfinite(AGB_masked)]
    IQ_mask = (AGB_masked < np.quantile(AGB_masked, 0.80)) & (AGB_masked > np.quantile(AGB_masked, 0.20))
    positive_values = AGB_masked[IQ_mask][AGB_masked[IQ_mask] >= 0]
    negative_values = AGB_masked[IQ_mask][AGB_masked[IQ_mask] < 0]

    if len(positive_values)>0:
        pointx_pos, pointy_pos, _, _ = violins(positive_values, pos=i, spread=0.3, max_num_points=1000)
        ax[1,1].scatter(pointy_pos, pointx_pos, color='#d95f02', alpha=0.2, marker='.')
        
    if len(negative_values)>0:
        pointx_neg, pointy_neg, _, _ = violins(negative_values, pos=i, spread=0.3, max_num_points=1000)
        ax[1,1].scatter(pointy_neg, pointx_neg, color='#7570b3', alpha=0.2, marker='.')

    # Plot the mean as a large diamond
    ax[1,1].scatter(i, np.nanquantile(AGB_masked[IQ_mask], 0.5), marker='d', s=200, color='black', alpha=0.5)
    
    mean_agb[j] = np.nanquantile(AGB_masked[IQ_mask], 0.5)
    i+=1

# Set the x-ticks and x-tick labels
ax[1,1].set_xticks(np.arange(len(list(AgePartition.keys()))))
ax[1,1].set_xticklabels(list(AgePartition.keys()), rotation=90, size=14)

# Rest of your plot settings
ax[1,1].set_ylabel('AGC changes x (-1) [gC m$^{-2}$ year$^{-1}$]', size=14)
ax[1,1].spines['top'].set_visible(False)
ax[1,1].spines['right']. set_visible(False)
ax[1,1].text(0.05, 1.1, 'd', transform=ax[1,1].transAxes, fontsize=16, fontweight='bold', va='top')
ax[1,1].tick_params(labelsize=12)
ax[1,1].set_ylim(-300, 800)

# Annotate for 'Carbon Source' above 0
ax[1,1].annotate('Carbon loss', xy=(0.4, 50), xytext=(0.4, 350),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=2),
                  ha='center', va='bottom', color='#d95f02', fontweight= 'bold', fontsize=14)

# Annotate for 'Carbon Sink' below 0
ax[1,1].annotate('Carbon gain', xy=(0.4, -50), xytext=(0.4, -280),
                  arrowprops=dict(facecolor='red', arrowstyle="<-", linewidth=2),
                  ha='center', va='bottom', color='#7570b3', fontweight= 'bold', fontsize=14)
ax[1,1].axhline(y=0, c='red', linestyle='dashed', linewidth=2)


plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/fig3.png', dpi=300)



