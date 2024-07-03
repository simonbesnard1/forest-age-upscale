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

#%% Load management data
forest_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ForestFraction_1deg').forest_fraction
management_fraction = xr.open_zarr('/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-1.0/ManagementTypeFrac_1deg').where(forest_fraction >0)

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
            #NEE_2010_filtered = apply_blur(NEE_2010.where(np.isfinite(NEE_2010), 0))
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
            #NEE_2020_filtered = apply_blur(NEE_2020.where(np.isfinite(NEE_2020), 0))
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
    #NEE_2010_filtered = apply_blur(NEE_2010)
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
    #NEE_2020_filtered = apply_blur(NEE_2020)

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
plt.savefig('/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/figS15.png', dpi=300)
