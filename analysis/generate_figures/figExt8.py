import xarray as xr
import numpy as np
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
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
    'legend.fontsize': 12,
    'text.usetex': True,
}

mpl.rcParams.update(params)

# Define function to compute area-weighted aggregation
def area_weighted_mean(data, lat_factor, lon_factor):
    """
    Computes area-weighted mean for coarser spatial windows.
    Parameters:
        data (xarray.DataArray): The data to aggregate
        lat_factor (int): Number of grid cells to aggregate in latitude
        lon_factor (int): Number of grid cells to aggregate in longitude
    Returns:
        xarray.DataArray: Area-weighted mean aggregated data
    """
    # Compute latitude weights (cosine of latitude in radians)
    lat_weights = np.cos(np.deg2rad(data.latitude))
    
    # Expand dims to match data shape
    weights = lat_weights.broadcast_like(data)
    
    # Compute weighted sum and normalize
    weighted_sum = (data * weights).coarsen(latitude=lat_factor, longitude=lon_factor, boundary="trim").sum()
    weight_sum = weights.coarsen(latitude=lat_factor, longitude=lon_factor, boundary="trim").sum()
    
    return weighted_sum / weight_sum  # Normalize by total weight


#%% Specify data and plot directories
data_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/data/Age_upscale_100m'
plot_dir = '/home/simon/Documents/science/research_paper/global_age_Cdyn/figs/'

#%% Load forest fraction
forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction

#%% Load stand-replaced / aging agb difference
forest_fraction = forest_fraction = xr.open_zarr(os.path.join(data_dir,'ForestFraction_1deg')).forest_fraction

#%% Calculate total per area
AgeDiffPartition_fraction_1deg =  xr.open_zarr(os.path.join(data_dir,"AgeDiffPartition_1deg"))

OG_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '>200')
OG_stand_replaced_class = OG_stand_replaced_class.where(forest_fraction>.2)

young_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '0-20')
young_stand_replaced_class = young_stand_replaced_class.where(forest_fraction>.2)

maturing_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '20-80')
maturing_stand_replaced_class = maturing_stand_replaced_class.where(forest_fraction>.2)

mature_stand_replaced_class = AgeDiffPartition_fraction_1deg.stand_replaced_class_partition.sel(age_class = '80-200')
mature_stand_replaced_class = mature_stand_replaced_class.where(forest_fraction>.2)


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
nee_changes = []
valid_model_= []
for member_ in NEE_RECCAP.ensemble_member:
    NEE_2010 =  NEE_RECCAP.sel(time= slice('2009-01-01', '2011-12-31')).isel(ensemble_member= member_).mean(dim='time') * 1e+15
    if not np.isnan(NEE_2010.values).all():
        NEE_2010 = NEE_2010 - lateral_fluxes_sink_2010 + lateral_fluxes_source_2010
        NEE_2010 = NEE_2010.where(forest_fraction>.2)
        NEE_2010_filtered = filter_nan_gaussian_conserving(NEE_2010, length_km=500)
        NEE_2010_filtered = NEE_2010_filtered.reindex(latitude=NEE_2010_filtered.latitude[::-1])
        NEE_2010_filtered['latitude'] = forest_fraction['latitude']
        NEE_2010_filtered['longitude'] = forest_fraction['longitude']
        NEE_2010_filtered = NEE_2010_filtered.where(forest_fraction>.2)
        
        NEE_2020 =  NEE_RECCAP.sel(time= slice('2019-01-01', '2021-12-31')).isel(ensemble_member= member_).mean(dim='time') * 1e+15
        NEE_2020 = NEE_2020 - lateral_fluxes_sink_2020 + lateral_fluxes_source_2020
        NEE_2020 = NEE_2020.where(forest_fraction>.2)
        NEE_2020_filtered = filter_nan_gaussian_conserving(NEE_2020, length_km=500)
        NEE_2020_filtered = NEE_2020_filtered.reindex(latitude=NEE_2020_filtered.latitude[::-1])
        NEE_2020_filtered['latitude'] = forest_fraction['latitude']
        NEE_2020_filtered['longitude'] = forest_fraction['longitude']
        NEE_2020_filtered = NEE_2020_filtered.where(forest_fraction>.2)
        
        nee_change = NEE_2020_filtered - NEE_2010_filtered

        nee_changes.append(nee_change)
        valid_model_.append(model_names[int(member_)])

# Convert NEE changes to an Xarray DataArray
nee_changes = xr.concat(nee_changes, dim="ensemble_member")

# Define spatial scales
coarse_scales = [1, 2, 5, 10]

# Create a list to store stacked arrays for each scale
stacked_data = dict()

# Iterate over spatial scales
for scale in coarse_scales:
    lat_factor = scale
    lon_factor = scale
    
    # Create an empty list for stacking per scale
    scale_data = []

    # Iterate over NEE members and Stand-replacement members
    for nee_idx in range(9):  # 9 NEE members
        member_data_list = []  # Store all stand members for this NEE member
        
        for stand_idx in range(20):  # 20 stand members
            
            # Select specific NEE and stand-replacement members
            OG_stand_replacement_coarse = area_weighted_mean(
                OG_stand_replaced_class.sel(members=stand_idx), lat_factor, lon_factor
            ).where(lambda x: x > 0)  # Ensure values >0
            
            # Select specific NEE and stand-replacement members
            young_stand_replacement_coarse = area_weighted_mean(
                young_stand_replaced_class.sel(members=stand_idx), lat_factor, lon_factor
            ).where(lambda x: x > 0)  # Ensure values >0
            
            
            maturing_stand_replacement_coarse = area_weighted_mean(
                maturing_stand_replaced_class.sel(members=stand_idx), lat_factor, lon_factor
            ).where(lambda x: x > 0)  # Ensure values >0
            
            
            mature_stand_replacement_coarse = area_weighted_mean(
                mature_stand_replaced_class.sel(members=stand_idx), lat_factor, lon_factor
            ).where(lambda x: x > 0)  # Ensure values >0
            
            nee_changes_coarse = area_weighted_mean(
                nee_changes.sel(ensemble_member=nee_idx), lat_factor, lon_factor
            )

            # Create a dataset for this member combination
            member_data = xr.Dataset({
                "OG_stand_replacement_extent": OG_stand_replacement_coarse.drop_vars('age_class'),
                "young_stand_replacement_extent": young_stand_replacement_coarse.drop_vars('age_class'),
                "maturing_stand_replacement_extent": maturing_stand_replacement_coarse.drop_vars('age_class'),
                "mature_stand_replacement_extent": mature_stand_replacement_coarse.drop_vars('age_class'),
                "nee_change": nee_changes_coarse
            })
            
            # Assign stand member coordinate
            member_data = member_data.expand_dims({"stand_member": [stand_idx]})
            member_data_list.append(member_data)
        
        # Stack all stand members into a single dataset along "stand_member"
        stand_dataset = xr.concat(member_data_list, dim="stand_member")
        
        # Assign NEE member coordinate
        stand_dataset = stand_dataset.expand_dims({"nee_member": [valid_model_[nee_idx]]})
        scale_data.append(stand_dataset)

    # Stack all NEE members into a single dataset along "nee_member"
    scale_dataset = xr.concat(scale_data, dim="nee_member")
    
    # Assign scale coordinate
    scale_dataset = scale_dataset.assign_coords(scale=scale)
    
    # Append to the final stacked list
    stacked_data.update({str(scale):scale_dataset})

#%% Plot data

# Define the scale to plot
selected_scale = "5"  # Example: 5°×5° resolution

# Load precomputed median dataset for selected scale
median_dataset = stacked_data[selected_scale].median(dim=["nee_member", 'stand_member'])


age_classes = {
    "Young Forests (0-20 yrs)": "young_stand_replacement_extent",
    "Maturing Forests (21-80 yrs)": "maturing_stand_replacement_extent",
    "Mature Forests (81-200 yrs)": "mature_stand_replacement_extent",
    "Old Forests ($>$200 yrs)": "OG_stand_replacement_extent"
}

#%% Create figure

# Define subplot grid (2x2) and create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 11), gridspec_kw={'wspace': 0, 'hspace': 0.1}, constrained_layout=True)

# Define subplot labels
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

# Loop through the age classes, assigning each to a subplot
for (i, (age_label, age_var)), (row, col) in zip(enumerate(age_classes.items()), itertools.product(range(2), range(2))):
    ax = axes[row, col]  
    # Extract median values for regression
    x_values = median_dataset[age_var].values.flatten()
    y_values = median_dataset["nee_change"].values.flatten()
    
    # Remove NaNs and filter values
    valid_mask = ~np.isnan(x_values) & ~np.isnan(y_values) & (x_values > 0.001)
    x_values = (x_values[valid_mask] /10) * 100
    y_values = y_values[valid_mask]
    
    # Perform regression on median values
    slope_full, intercept_full, r_value, p_value_full, std_err = stats.linregress(x_values, y_values)
    
    # Load full dataset for Jackknife Analysis
    full_dataset = stacked_data[selected_scale]
    
    # Jackknife Resampling: Exclude one NEE inversion product at a time
    unique_nee_members = valid_model_  # We have 9 NEE members
    jackknife_slopes = []
    jackknife_intercepts = []
    
    for excluded_member in unique_nee_members:
        # Exclude the selected NEE member
        jackknife_dataset = full_dataset.sel(nee_member=[i for i in unique_nee_members if i != excluded_member])
        
        # Compute median excluding that NEE member
        jackknife_median = jackknife_dataset.median(dim=["nee_member", "stand_member"])
        
        # Extract values for regression
        x_jack = jackknife_median[age_var].values.flatten()
        y_jack = jackknife_median["nee_change"].values.flatten()
    
        # Remove NaNs and filter
        valid_mask = ~np.isnan(x_jack) & ~np.isnan(y_jack) & (x_jack > 0.001)
        x_jack = (x_jack[valid_mask] /10) * 100
        y_jack = y_jack[valid_mask]
    
        # Compute regression
        slope, intercept, _, _, _ = stats.linregress(x_jack, y_jack)
        jackknife_slopes.append(slope)
        jackknife_intercepts.append(intercept)
    
    # Compute confidence intervals from Jackknife estimates
    slope_mean = np.mean(jackknife_slopes)
    slope_std = np.std(jackknife_slopes)
    intercept_mean = np.mean(jackknife_intercepts)
    intercept_std = np.std(jackknife_intercepts)
    
    slope_lower = slope_mean - 1.96 * slope_std  # 95% CI
    slope_upper = slope_mean + 1.96 * slope_std
    
    intercept_lower = intercept_mean - 1.96 * intercept_std
    intercept_upper = intercept_mean + 1.96 * intercept_std
    
    # Generate regression line for plotting
    x_range = np.linspace(min(x_values), max(x_values), 100)
    y_full = intercept_full + slope_full * x_range
    y_lower = intercept_lower + slope_lower * x_range
    y_upper = intercept_upper + slope_upper * x_range
    
    # Define colors
    color_positive = "#d95f02"  # Orange for positive slopes
    color_negative = "#7570b3"  # Blue for negative slopes
    
    # Plot individual NEE member regression lines (background)
    positive_slopes = 0
    negative_slopes = 0
    significance_level = 0.05
    non_significant_slopes = 0
    for nee_idx in unique_nee_members:
        member_dataset = full_dataset.sel(nee_member=nee_idx).median(dim="stand_member")
    
        # Extract values for regression
        x_member = member_dataset[age_var].values.flatten() #/ member_dataset["OG_class_extent"].values.flatten()
        y_member = member_dataset["nee_change"].values.flatten()
    
        # Remove NaNs
        valid_mask = ~np.isnan(x_member) & ~np.isnan(y_member) & (x_member > 0.001)
        x_member = (x_member[valid_mask] /10) * 100
        y_member = y_member[valid_mask]
    
        # Compute regression for individual NEE member
        slope, intercept, _, p_value, _ = stats.linregress(x_member, y_member)
    
        # Generate regression line
        y_member_reg = intercept + slope * x_range
    
        # Plot individual regression line
        ax.plot(x_range, y_member_reg, color="grey", linestyle="dashed", alpha=0.3, linewidth=2)
        
        # Annotate model name at the end of the regression line
        ax.text(x_range[-1]+0.001, y_member_reg[-1], nee_idx, fontsize=10, color="grey", alpha=0.7, ha="left", va="center")
        
        if p_value < significance_level:
            if slope > 0:
                positive_slopes += 1
            else:
                negative_slopes += 1
        else:
            non_significant_slopes += 1
    
    ax.text(0.05, 0.30, f'(+)slope = {positive_slopes} members', 
                 transform=ax.transAxes, verticalalignment='top', 
                 fontsize=12, color="#d95f02", fontweight='bold')
    
    ax.text(0.055, 0.37, f'(–)slope = {negative_slopes} members',  # en dash
                 transform=ax.transAxes, verticalalignment='top', 
                 fontsize=12, color='#7570b3', fontweight='bold')
    
    # Scatter plot using median values per spatial window
    positive_indices = y_values >= 0
    negative_indices = y_values < 0
    ax.scatter(x_values[positive_indices], y_values[positive_indices], s=50, color=color_positive, alpha=0.7)
    ax.scatter(x_values[negative_indices], y_values[negative_indices], s=50, color=color_negative, alpha=0.7)
    
    # Plot median regression line (bold) and confidence interval
    ax.plot(x_range, y_full, color='black', linewidth=2, label="Median Regression Fit")
    ax.fill_between(x_range, y_lower, y_upper, color="grey", alpha=0.3, label="95% Jackknife CI")
    
    # Annotate regression statistics
    ax.text(.05, 0.08, f'R² = {r_value**2:.2f}\nSlope = {slope_full:.2f} gC m$^{{-2}}$ year$^{{-1}}$ per % year$^{{-1}}$\np-value = {p_value_full:.3f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black'))
    
    # axes labels and formatting
    ax.set_ylabel('Changes in NEE [gC m$^{-2}$ year$^{-1}$]', size=14)
    ax.set_xlabel('Stand-replacement extent [% year$^{-1}$]', size=14)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    
    # # Add annotations
    # ax.annotate(r'$\uparrow$C source or $\downarrow$C sink  ', xy=(.065, 2), xytext=(.065, 20),
    #             arrowprops=dict(facecolor=color_positive, arrowstyle="<-", linewidth=2),
    #             ha='center', va='bottom', color=color_positive, fontweight='bold', fontsize=13)
    
    # ax.annotate(r'$\uparrow$C sink', xy=(.065, -2), xytext=(.065, -18),
    #             arrowprops=dict(facecolor=color_negative, arrowstyle="<-", linewidth=2),
    #             ha='center', va='bottom', color=color_negative, fontweight='bold', fontsize=13)
    
    ax.axhline(y=0, c='red', linestyle='dashed', linewidth=2)  # Add zero line
    
    # Panel Label
    ax.set_title(f"{age_label}", fontsize=14)
    
    # Panel Label
    ax.text(0.05, 1.05, subplot_labels[i], transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top')
# Show plot
plt.savefig(os.path.join(plot_dir,'figExt8.png'), dpi=300)
