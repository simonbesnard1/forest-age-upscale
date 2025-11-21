#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset GAMI v3 for each GeoJSON ROI, plot map + histogram, and save to file.
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import glob
import matplotlib as mpl
params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times', #'cmr10',
    'font.size': 16,
    # axes
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 14,
    # tex
    'text.usetex': True,
}

mpl.rcParams.update(params)


# -------------------------------------------------------------------
# Paths & settings
# -------------------------------------------------------------------
GAMI_ZARR = "/home/simon/hpc_group/scratch/besnard/upscaling/Age_upscale_100m/XGBoost/version-3.0/AgeUpscale_100m"
ABG_ZARR = "/home/simon/hpc_home/projects/forest-age-upscale/data/cubes/ESACCI_BIOMASS_100m_v6"
ROI_GEOJSON_GLOB = "/home/simon/Documents/science/GFZ/projects/foreststrucflux/data/geojson/*.geojson"
OUT_DIR = "/home/simon/Documents/science/GFZ/projects/foreststrucflux/figures/gami_rois"

# name of the variable in the GAMI dataset (change if needed)
VAR_NAME = "forest_age"  # or e.g. "age_2010", "age", etc.


def load_gami(path: str) -> xr.Dataset:
    ds = xr.open_zarr(path)
    # Ensure CRS is set for rioxarray; adjust if your CRS is different
    if not ds.rio.crs:
        ds = ds.rio.write_crs("EPSG:4326", inplace=False)
    return ds


def subset_ds_to_roi(ds: xr.Dataset, roi_gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """
    First subset by bbox, then clip by polygon. Assumes lon/lat named like 'longitude'/'latitude'.
    """
    geom = roi_gdf.to_crs("EPSG:4326").geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds

    # quick bbox subset (names might be 'lon'/'lat' in your ds; adapt if needed)
    lon_name = "longitude"
    lat_name = "latitude"

    ds_bbox = ds.sel(
        {lon_name: slice(minx, maxx),
         lat_name: slice(maxy, miny)}
    ).mean(dim = ("time","members"))
    
    return ds_bbox


def subset_abg_to_roi(ds_abg: xr.Dataset, roi_gdf: gpd.GeoDataFrame) -> xr.Dataset:
    geom = roi_gdf.to_crs("EPSG:4326").geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds

    lon_name, lat_name = "longitude", "latitude"

    ds_bbox = ds_abg.sel(
        {lon_name: slice(minx, maxx),
         lat_name: slice(maxy, miny)}
    )   
    return ds_bbox * 0.47


def compute_growth_slope(ds_abg_roi: xr.Dataset) -> xr.DataArray:
    """
    Compute linear biomass trend (2007–2022) per pixel.
    Returns a DataArray matching lat/lon with 'growth' values.
    """

    abg = ds_abg_roi["aboveground_biomass"]  # (lat, lon, time)
    t = (abg["time"].dt.year - abg["time"].dt.year.min()).astype(float)

    # slope = cov(x,y)/var(x) in a vectorized form
    x = t - t.mean()
    y = abg - abg.mean("time")

    cov = (x * y).sum("time")
    var = (x**2).sum()

    slope = cov / var
    slope = slope.rename("biomass_growth")

    return slope

def plot_map_hist_growth_biomass(
    ds_roi_age: xr.Dataset,
    da_growth: xr.DataArray,
    da_biomass_2020: xr.DataArray,
    var_name: str,
    title: str,
    out_path: Path,
):
    """
    Four-panel figure:
      (1) Age map
      (2) Age histogram
      (3) Age – biomass growth (trend)
      (4) Age – biomass at 2020-01-01
    """

    da_age = ds_roi_age[var_name]

    # Common mask where all three have valid values
    common_mask = (
        np.isfinite(da_age.values)
        & np.isfinite(da_growth.values)
        & np.isfinite(da_biomass_2020.values)
    )

    if not np.any(common_mask):
        print(f"[WARN] No valid data for {title}, skipping plot.")
        return

    age_flat = da_age.values[common_mask]
    growth_flat = da_growth.values[common_mask]
    biomass_flat = da_biomass_2020.values[common_mask]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # ------------------------------------------------------------------
    # (1) Age map
    # ------------------------------------------------------------------
    ax_map = axes[0, 0]
    da_age.plot.imshow(ax=ax_map, cmap="viridis")
    ax_map.set_title("Forest age (map)")

    # ------------------------------------------------------------------
    # (2) Age histogram
    # ------------------------------------------------------------------
    ax_hist = axes[0, 1]
    ax_hist.hist(age_flat, bins=40, color="grey")
    ax_hist.set_title("Age distribution")
    ax_hist.set_xlabel("Age [years]")
    ax_hist.set_ylabel("Count")
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)


    # ------------------------------------------------------------------
    # (3) Age vs biomass growth
    # ------------------------------------------------------------------
    ax_sc1 = axes[1, 0]
    ax_sc1.scatter(age_flat, growth_flat, s=1, alpha=0.4)
    ax_sc1.set_title("Age vs biomass trend (2007–2022)")
    ax_sc1.set_xlabel("Age [years]")
    ax_sc1.set_ylabel("Biomass trend [MgC/ha/year]")
    ax_sc1.spines['top'].set_visible(False)
    ax_sc1.spines['right'].set_visible(False)


    # ------------------------------------------------------------------
    # (4) Age vs biomass at 2020-01-01
    # ------------------------------------------------------------------
    ax_sc2 = axes[1, 1]
    ax_sc2.scatter(age_flat, biomass_flat, s=1, alpha=0.4)
    ax_sc2.set_title("Age vs biomass (2020-01-01)")
    ax_sc2.set_xlabel("Age [years]")
    ax_sc2.set_ylabel("Biomass [MgC/ha]")
    ax_sc2.spines['top'].set_visible(False)
    ax_sc2.spines['right'].set_visible(False)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


ds_age = load_gami(GAMI_ZARR)
ds_abg = xr.open_zarr(ABG_ZARR)

roi_files = glob.glob(ROI_GEOJSON_GLOB)

for roi_path in roi_files:
    roi_path = Path(roi_path)
    roi_name = roi_path.stem
    print(f"Processing ROI: {roi_name}")

    gdf = gpd.read_file(roi_path)

    # subset age (GAMI)
    ds_roi_age = subset_ds_to_roi(ds_age, gdf)

    # subset biomass cube
    ds_abg_roi = subset_abg_to_roi(ds_abg, gdf)

    # compute biomass growth slope
    growth_da = compute_growth_slope(ds_abg_roi)

    # biomass at 2020-01-01
    abg_2020 = ds_abg_roi["aboveground_biomass"].sel(time="2020-01-01")
    abg_2020 = abg_2020.rename("biomass_2020")

    # plot 4-panel figure
    out_file = Path(OUT_DIR) / f"gamiv2_{VAR_NAME}_{roi_name}_growth_biomass.png"
    title = f"{roi_name} – Age, growth and biomass"
    plot_map_hist_growth_biomass(ds_roi_age, growth_da, abg_2020, VAR_NAME, title, out_file)

    print(f"  -> saved {out_file}")


