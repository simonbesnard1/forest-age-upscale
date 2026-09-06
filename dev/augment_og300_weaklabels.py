#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2025 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2025 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2
# File    :   augment_og300_weaklabels.py

Augments the OG-300 ("definitely old growth") training class with weakly
labelled examples mined from the global cubes already used operationally.

Why: the field-reference OG-300 class has almost no low-biomass examples
(p10 AGB ~155 Mg/ha), so the old-growth classifier never learns that low
biomass does not rule out old growth -- see the low-biomass-old-forest
mismapping investigation in the forest-age-upscale project. Sourcing new
field plots for naturally low-biomass old-growth biomes (permafrost taiga,
dry woodland, silvopastoral savanna) is not something we can do here.
Instead, this mines pixels that are themselves strong biomass-independent
evidence of old growth: forest with NO disturbance detected anywhere in the
full Landsat monitoring record (LTSD == 50, the right-censored sentinel),
specifically the low-biomass tail of that population in a handful of
biome-diverse regions.

These are weak labels, not field-verified ages: LTSD==50 only guarantees
"undisturbed for at least the ~50y record", not "definitely 300 years old".
They are added purely as auxiliary OG-300 classifier training rows (age is
set to the existing 300 sentinel, which the regressor already excludes via
its own age<300 filter -- see MLData.get_y), to widen the classifier's
exposure to the "old + naturally low biomass" combination beyond what
re-weighting the existing ~40 real low-biomass OG-300 plots can achieve
(see ageUpscaling/utils/oversampling.py for that complementary fix).

Usage
-----
    python dev/augment_og300_weaklabels.py

Writes a new file, `<original>_augmented_weaklabel.nc`, alongside the
original training dataset. The original file is never modified. Point
`training_dataset:` in the data_config yaml at the new file to use it.
"""
import numpy as np
import xarray as xr

TRAINING_DATASET = "/home/simon/hpc_home/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_subsetFIA_v7.nc"
CUBES = "/home/simon/hpc_home/projects/forest-age-upscale/data/cubes"
OUT_PATH = TRAINING_DATASET.replace(".nc", "_augmented_weaklabel.nc")

END_YEAR = "2020-01-01"
OG_SENTINEL_AGE = 300

# Biome-diverse regions known (from prior diagnostics in this project) to
# hold a substantial population of long-undisturbed, naturally low-biomass
# forest/woodland -- exactly the population missing from the OG-300 class.
REGIONS = [
    # name, lat_hi, lat_lo, lon_lo, lon_hi
    ("Siberia_larch_taiga",        63.6,  63.5, 105.3, 105.4),
    ("Miombo_dry_woodland_Zambia", -14.1, -14.2, 28.1,  28.2),
    ("Mediterranean_dehesa_Spain",  39.4,  39.3, -5.9,  -5.8),
    ("Canada_treeline",             60.0,  59.0, -105.0, -104.0),
    ("Australian_woodland",        -24.0, -25.0, 135.0, 136.0),
]

# Fraction of each region's LTSD==50 AGB distribution to draw from (the low
# tail -- this is deliberately the population the classifier currently
# misses).
LOW_AGB_QUANTILE = 0.33
MAX_SAMPLES_PER_REGION = 200
RANDOM_SEED = 42

WORLDCLIM_FEATURES = [
    "AnnualMeanTemperature_WorlClim", "AnnualPrecipitation_WorlClim", "AnnualSrad_WorlClim",
    "AnnualVapr_WorlClim", "AnnualWind_WorlClim", "Isothermality_WorlClim",
    "MaxTemperatureofWarmestMonth_WorlClim", "MeanDiurnalRange_WorlClim",
    "MeanTemperatureofColdestQuarter_WorlClim", "MeanTemperatureofDriestQuarter_WorlClim",
    "MeanTemperatureofWarmestQuarter_WorlClim", "MeanTemperatureofWettestQuarter_WorlClim",
    "MinTemperatureofColdestMonth_WorlClim", "PrecipitationSeasonality_WorlClim",
    "PrecipitationofColdestQuarter_WorlClim", "PrecipitationofDriestMonth_WorlClim",
    "PrecipitationofDriestQuarter_WorlClim", "PrecipitationofWarmestQuarter_WorlClim",
    "PrecipitationofWettestMonth_WorlClim", "PrecipitationofWettestQuarter_WorlClim",
    "TemperatureAnnualRange_WorlClim", "TemperatureSeasonality_WorlClim",
]


def mine_region(name, lat_hi, lat_lo, lon_lo, lon_hi, ltsd_cube, agb_cube, canopy_cube, clim_cube, rng):
    IN = {"latitude": slice(lat_hi, lat_lo), "longitude": slice(lon_lo, lon_hi)}
    buf = 0.02
    BUFFER_IN = {
        "latitude": slice(lat_hi + buf, lat_lo - buf),
        "longitude": slice(lon_lo - buf, lon_hi + buf),
    }

    # Merge by coordinate label (not raw .values reshaping): these are three
    # independently-opened zarr stores, and while they're all nominally the
    # same 100m global grid, xr.merge's label-based alignment is the safe
    # way to guarantee the three arrays line up pixel-for-pixel rather than
    # assuming identical float coordinate arrays across separate stores.
    ltsd_sub = ltsd_cube.sel(time=END_YEAR).sel(IN)[["LandsatDisturbanceTime"]]
    agb_sub = agb_cube.sel(time=END_YEAR).sel(IN)[["aboveground_biomass"]]
    canopy_sub = canopy_cube.sel(time=END_YEAR).sel(IN)[["canopy_height"]]
    merged = xr.merge([ltsd_sub, agb_sub, canopy_sub], join="inner", compat="override").compute()

    ltsd_flat = merged["LandsatDisturbanceTime"].values.reshape(-1)
    agb_flat = merged["aboveground_biomass"].values.reshape(-1)
    canopy_flat = merged["canopy_height"].values.reshape(-1)
    lat2d, lon2d = np.meshgrid(merged.latitude.values, merged.longitude.values, indexing="ij")
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)

    mask_forest = np.isfinite(agb_flat) & (agb_flat > 0)
    mask_undisturbed = np.isfinite(ltsd_flat) & (ltsd_flat == 50)
    mask = mask_forest & mask_undisturbed & np.isfinite(canopy_flat) & (canopy_flat > 0)

    if mask.sum() == 0:
        print(f"  {name}: no candidate pixels, skipping")
        return None

    agb_thresh = np.nanquantile(agb_flat[mask], LOW_AGB_QUANTILE)
    mask_low = mask & (agb_flat <= agb_thresh)

    n_candidates = mask_low.sum()
    n_take = min(n_candidates, MAX_SAMPLES_PER_REGION)
    idx = np.where(mask_low)[0]
    idx = rng.choice(idx, size=n_take, replace=False)

    print(f"  {name}: {n_candidates} low-AGB (<={agb_thresh:.1f} Mg/ha) undisturbed candidates, "
          f"sampling {n_take}")

    sel_lat = lat_flat[idx]
    sel_lon = lon_flat[idx]
    sel_agb = agb_flat[idx].astype("float32")
    sel_canopy = canopy_flat[idx].astype("float32")

    # Climate: interpolate the 1km WorldClim grid to the exact sampled points
    clim_at_points = clim_cube[WORLDCLIM_FEATURES].sel(
        latitude=xr.DataArray(sel_lat, dims="points"),
        longitude=xr.DataArray(sel_lon, dims="points"),
        method="nearest",
    )

    data = {
        "age": ("sample", np.full(n_take, OG_SENTINEL_AGE, dtype="float64")),
        "agb": ("sample", sel_agb.astype("float64")),
        "agb_gapfilled": ("sample", sel_agb.astype("float64")),
        "canopy_height_gapfilled": ("sample", sel_canopy.astype("float64")),
        "weak_label_source": ("sample", np.full(n_take, name)),
    }
    for var in WORLDCLIM_FEATURES:
        data[var] = ("sample", clim_at_points[var].values.astype("float64"))

    ds = xr.Dataset(data, coords={"sample": np.arange(n_take)})
    return ds


def main():
    rng = np.random.RandomState(RANDOM_SEED)

    print("Opening cubes...")
    ltsd_cube = xr.open_zarr(f"{CUBES}/LandsatDisturbanceTime_100m")
    ltsd_cube = ltsd_cube.where(ltsd_cube >= 1)
    agb_cube = xr.open_zarr(f"{CUBES}/ESACCI_BIOMASS_100m_v6")
    canopy_cube = xr.open_zarr(f"{CUBES}/canopyHeight_potapov_100m")
    clim_cube = xr.open_zarr(f"{CUBES}/WorlClim_1km")

    print("Mining weak-labelled low-biomass old-growth candidates per region...")
    region_datasets = []
    for name, lat_hi, lat_lo, lon_lo, lon_hi in REGIONS:
        ds = mine_region(name, lat_hi, lat_lo, lon_lo, lon_hi, ltsd_cube, agb_cube, canopy_cube, clim_cube, rng)
        if ds is not None:
            region_datasets.append(ds)

    if not region_datasets:
        raise RuntimeError("No weak-labelled samples were mined from any region.")

    max_samples = max(ds.sizes["sample"] for ds in region_datasets)

    print("\nLoading original training dataset...")
    original = xr.open_dataset(TRAINING_DATASET)
    next_cluster = int(original.cluster.values.max()) + 1

    new_clusters = []
    for i, ds in enumerate(region_datasets):
        ds = ds.reindex(sample=np.arange(max_samples))
        ds = ds.expand_dims({"cluster": [next_cluster + i]})
        new_clusters.append(ds)

    augmented_new = xr.concat(new_clusters, dim="cluster")

    print(f"\nMerging {augmented_new.sizes['cluster']} new weak-labelled clusters "
          f"({sum(ds.sizes['sample'] for ds in region_datasets)} total samples) "
          f"with {original.sizes['cluster']} original clusters...")

    combined = xr.concat([original, augmented_new], dim="cluster", join="outer", combine_attrs="override")
    combined.attrs = dict(original.attrs)
    combined.attrs["weak_label_augmentation"] = (
        f"{sum(ds.sizes['sample'] for ds in region_datasets)} weak-labelled OG-300 samples added "
        f"from {len(region_datasets)} regions (LTSD==50, bottom {LOW_AGB_QUANTILE:.0%} AGB quantile); "
        "see dev/augment_og300_weaklabels.py"
    )

    print(f"Writing augmented dataset to {OUT_PATH} ...")
    combined.to_netcdf(OUT_PATH)

    print("\n=== Summary ===")
    for name, ds in zip([r[0] for r in REGIONS], region_datasets):
        agb_vals = ds["agb"].values
        print(f"  {name}: n={ds.sizes['sample']}  AGB median={np.nanmedian(agb_vals):.1f}  "
              f"p90={np.nanpercentile(agb_vals, 90):.1f} Mg/ha")
    print(f"\nOriginal clusters: {original.sizes['cluster']}  ->  augmented: {combined.sizes['cluster']}")
    print(f"Done. Point 'training_dataset:' at {OUT_PATH} to use the augmented data.")


if __name__ == "__main__":
    main()
