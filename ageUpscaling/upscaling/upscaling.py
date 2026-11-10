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
# File: upscaling.py

A method class for upscaling forest age ML model.

"""
import os
import shutil
from tqdm import tqdm
from itertools import product
from abc import ABC
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import numpy as np
import yaml as yml
import pickle

import geopandas as gpd
from rasterio.features import geometry_mask
import rioxarray as rio

import xarray as xr
import zarr
import dask
import dask.array as da
from shapely.geometry import Polygon
import pandas as pd

from sklearn.model_selection import train_test_split
import xgboost as xgb

from ageUpscaling.core.cube import DataCube
from ageUpscaling.transformers.spatial import interpolate_worlClim, block_mean_regrid
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.methods.xgboost import XGBoost
from ageUpscaling.methods.RandomForest import RandomForest
from ageUpscaling.methods.feature_selection import FeatureSelection
from ageUpscaling.methods.AgeFusion import AgeFusion, reproject_ltsd_to_year
from ageUpscaling.methods.extrapolation_index import ExtrapolationIndex

class UpscaleAge(ABC):
    """A class for upscaling forest age ML models.

    Parameters
    ----------
    DataConfig_path : str
        Path to the data configuration file.
    upscaling_config_path : str
        Path to the upscaling configuration file.
    base_dir : str
        Base directory for the study.
    algorithm : str, optional
        The algorithm to use (default is 'MLP').
    exp_name : str, optional
        The experiment name.
    study_dir : str, optional
        Directory for restoring the study.
    n_jobs : int, optional
        Number of workers (default is 1).
    """
    def __init__(self,
                 DataConfig_path: str,
                 upscaling_config_path: str,            
                 base_dir: str,
                 algorithm: str = 'MLP',
                 exp_name: str = None,
                 study_dir: str = None,
                 n_jobs: int = 1,
                 **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        
        with open(upscaling_config_path, 'r') as f:
            self.upscaling_config =  yml.safe_load(f)
            
        intact_forest = gpd.read_file(self.DataConfig['intact_forest_df'])
        self.intact_tropical_forest = intact_forest[intact_forest['IFL_ID'].str.contains('|'.join(['SAM', 'SEA', 'AFR']))]
        
        self.algorithm = algorithm
        self.base_dir = base_dir
        self.exp_name = exp_name
        
        if study_dir is None:
            study_dir = self.version_dir(self.base_dir, self.exp_name, self.algorithm)
            os.makedirs(study_dir, exist_ok=False)
        else:
            if not os.path.exists(study_dir):
                raise ValueError(f'restore path does not exist:\n{study_dir}')

        self.study_dir = study_dir
        self.n_jobs = n_jobs       
        self.valid_fraction= self.DataConfig["valid_fraction"]
        self.feature_selection= self.DataConfig["feature_selection"]
        self.feature_selection_method= self.DataConfig["feature_selection_method"]      
        self.upscaling_config['cube_location'] =  self.upscaling_config['cube_location']
        
        self.task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
        sync_file_features = os.path.abspath(f"{study_dir}/features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        
        self.clim_cube  = xr.open_zarr(self.DataConfig['clim_cube'], synchronizer=self.sync_feature)
        self.LastTimeSinceDist_cube = xr.open_zarr(self.DataConfig['LastTimeSinceDist_cube'], synchronizer=self.sync_feature)
        self.CRparams_cube = xr.open_zarr(self.DataConfig['CRparams_cube'], synchronizer=self.sync_feature)
        # agb_cube and canopyHeight_cube are read directly from the remote
        # GFZ object store (see data_config yaml): plain HTTP fetches, so
        # the local zarr.ProcessSynchronizer (a file-lock for a shared
        # local store) doesn't apply and is omitted here.
        self.agb_cube = xr.open_zarr(self.DataConfig['agb_cube'])
        self.canopyHeight_cube = xr.open_zarr(self.DataConfig['canopy_height_cube'])

        # AGB ensemble members are no longer read from a separately
        # precomputed cube: they're drawn on the fly per prediction chunk
        # from agb_cube's own mean/std (see _predict_func), removing what
        # used to be two sequential steps (precompute member perturbations,
        # then predict per member) down to one. The seed is derived
        # per-chunk (see _member_rng) so results stay reproducible across
        # reruns of the same chunk.
        self.member_random_seed = int(self.DataConfig.get('member_random_seed', 42))

        # Pixel-based model extrapolation index (see ExtrapolationIndex):
        # fit lazily on first use in _predict_func, not here, since
        # __init__ also runs for model_training() where no trained model
        # exists yet to fit it from. Cached afterwards so it's fit once
        # per process, not once per chunk.
        self.extrapolation_index = None
        self.extrapolation_index_features = None

        self.upscaling_config['sync_file_path'] = os.path.abspath(f"{study_dir}/cube_out_sync_{self.task_id}.zarrsync") 
        self.upscaling_config['output_writer_params']['dims']['latitude']  = self.agb_cube.latitude.values
        self.upscaling_config['output_writer_params']['dims']['longitude'] = self.agb_cube.longitude.values
        self.upscaling_config['output_writer_params']['dims']['members'] =  self.upscaling_config['num_members']
        self.tmp_folder = os.path.join(self.upscaling_config['tmp_dir'], 'AgeUpscaling/')
        self.pred_cube = DataCube(cube_config = self.upscaling_config)
        self.pred_cube.init_variable(self.upscaling_config['cube_variables'], 
                                     njobs= len(self.upscaling_config['cube_variables'].keys()))
        
    def version_dir(self, 
                    base_dir: str,
                    exp_name:str,
                    algorithm: str) -> str:
        """Creates a new version of a directory by appending the version number.

        If the directory already exists, it will be renamed to include the version number before the new directory is created.
        
        Parameters
        ----------
        base_dir : str
            The base directory where the new version of the study directory will be created.
        exp_name : str
            The experiment name.
        algorithm : str
            The name of the algorithm.
            
        Returns
        -------
        str
            The full path to the new version of the study directory.
        """
        
        return self.increment_dir_version(base_dir,exp_name, algorithm)
    
    @staticmethod
    def increment_dir_version(base_dir: str,
                              exp_name:str,
                              algorithm:str) -> str:
        """Increments the version of a directory by appending the next available version number.

        Parameters
        ----------
        base_dir : str
            The base directory for the study.
        exp_name : str
            The experiment name.
        algorithm : str
            The name of the algorithm.
        
        Returns
        -------
        str
            The name of the new directory with the incremented version number.
        """
        if not os.path.isdir(os.path.join(base_dir, exp_name, algorithm)):
            os.makedirs(os.path.join(base_dir, exp_name, algorithm))
        
        dir_list = [d for d in os.listdir(os.path.join(base_dir, exp_name, algorithm)) if d.startswith("version")]
        
        dir_list.sort()
        
        if len(dir_list) == 0:
            version = "1.0"
        else:
            last_dir = dir_list[-1]
            
            _, version = last_dir.split("-")
            
            major, minor = version.split(".")
            major = int(major)
            minor = int(minor)
            minor += 1
            if minor >= 10:
                major += 1
                minor = 0
            version = f"{major}.{minor}"
        
        return f"{base_dir}/{exp_name}/{algorithm}/version-{version}"

    def _member_rngs(self, IN) -> list:
        """Deterministic, independent RNG streams for on-the-fly AGB member perturbation.

        Seeded from `member_random_seed` plus this chunk's spatial bounds,
        so re-running the same chunk reproduces the same members, while
        different chunks and different members within a chunk get
        statistically independent draws (via `SeedSequence.spawn`).

        Parameters
        ----------
        IN : dict
            The chunk's subsetting dictionary (same as passed to `_predict_func`).

        Returns
        -------
        list of np.random.Generator
            One independent generator per ensemble member.
        """
        lat0 = float(IN['latitude'].start)
        lon0 = float(IN['longitude'].start)
        seed_seq = np.random.SeedSequence([
            self.member_random_seed,
            int(round(lat0 * 1e6)) & 0xFFFFFFFF,
            int(round(lon0 * 1e6)) & 0xFFFFFFFF,
        ])
        return [np.random.default_rng(s) for s in seed_seq.spawn(self.upscaling_config['num_members'])]

    def _ensure_extrapolation_index(self) -> None:
        """Fit the pixel-based extrapolation index once, on first use.

        Uses the member-0 regressor as a stable reference for feature
        importance weighting: a single, fixed reference rather than a
        per-member fit, so all 20 members' pixels are scored against the
        same feature-space notion of "novel relative to training data"
        rather than 20 slightly different ones tied to each member's own
        (independently feature-selected) model.
        """
        if self.extrapolation_index is not None:
            return

        with open(self.study_dir + "/save_model/best_Regressor_run0.pickle", 'rb') as f:
            regressor_config = pickle.load(f)
        reference_model = regressor_config['best_model']
        reference_features = regressor_config['selected_features']

        training_ds = xr.open_dataset(self.DataConfig['training_dataset'])
        X_train = training_ds[reference_features].to_array().transpose(
            'cluster', 'sample', 'variable'
        ).values.reshape(-1, len(reference_features))
        X_train = X_train[np.all(np.isfinite(X_train), axis=1)]

        ei = ExtrapolationIndex()
        feature_weights = None
        if self.algorithm == "XGBoost":
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(len(X_train), size=min(2000, len(X_train)), replace=False)
            feature_weights = ei.compute_feature_weights(reference_model, X_train[sample_idx], reference_features)
        ei.fit(X_train, feature_weights=feature_weights)

        self.extrapolation_index = ei
        self.extrapolation_index_features = reference_features

    @dask.delayed
    def _predict_func(self,
                      IN) -> None:
        """Predict forest age using the trained ML models.

        Parameters
        ----------
        IN : dict
            Dictionary for subsetting the input data cubes.
        """
          
        lat_start, lat_stop = IN['latitude'].start, IN['latitude'].stop
        lon_start, lon_stop = IN['longitude'].start, IN['longitude'].stop
        buffer_IN = Polygon([(lon_start, lat_start), (lon_start, lat_stop),(lon_stop, lat_stop), (lon_stop, lat_start)]).buffer(0.01)
        buffer_IN = {'latitude': slice(buffer_IN.bounds[3], buffer_IN.bounds[1], None),
                    'longitude': slice(buffer_IN.bounds[0], buffer_IN.bounds[2], None)}
    
        subset_LastTimeSinceDist_cube = self.LastTimeSinceDist_cube.sel(time = self.DataConfig['end_year']).sel(buffer_IN)
        subset_LastTimeSinceDist_cube = subset_LastTimeSinceDist_cube.where(subset_LastTimeSinceDist_cube>=1)
        
        if not np.isnan(subset_LastTimeSinceDist_cube.to_array().values).all():
                            
            subset_clim_cube = self.clim_cube.sel(buffer_IN)[[x for x in self.DataConfig['features'] if "WorlClim" in x]].astype('float16')
            subset_clim_cube = interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_LastTimeSinceDist_cube).sel(IN)
            subset_CRparams_cube = self.CRparams_cube.sel(buffer_IN).astype('float16')
            subset_CRparams_cube = interpolate_worlClim(source_ds = subset_CRparams_cube, target_ds = subset_LastTimeSinceDist_cube).sel(IN)
            subset_LastTimeSinceDist = subset_LastTimeSinceDist_cube.sel(IN).to_array()
            LTSD_anchor_flat = subset_LastTimeSinceDist.values.reshape(-1)
            anchor_year = int(self.DataConfig['end_year'].split('-')[0])
            all_years = sorted({
                pd.to_datetime(t).year
                for t in self.upscaling_config['output_writer_params']['dims']['time']
            })
            LTSD_by_year = {anchor_year: LTSD_anchor_flat}
            for year in all_years:
                if year == anchor_year:
                    continue
                LTSD_by_year[year] = reproject_ltsd_to_year(
                    LTSD_anchor_flat, target_year=year, anchor_year=anchor_year
                )

            # Canopy height comes from the remote 30m Potapov store (see
            # data_config yaml): block-mean aggregate onto the prediction
            # grid rather than point-interpolate, since every native pixel
            # should contribute to the target cell it falls inside.
            canopy_feature_name = [x for x in self.DataConfig['features'] if "canopy_height" in x][0]
            canopy_times = self.upscaling_config['output_writer_params']['dims']['time']
            canopy_regridded = [
                block_mean_regrid(self.canopyHeight_cube['canopy_height'].sel(time=t), subset_LastTimeSinceDist_cube)
                .sel(IN)
                .expand_dims({'time': [pd.to_datetime(t)]})
                for t in canopy_times
            ]
            subset_canopyHeight_cube = xr.concat(canopy_regridded, dim='time').astype('float16').to_dataset(name=canopy_feature_name)
            subset_canopyHeight_cube = subset_canopyHeight_cube.where(subset_canopyHeight_cube > 0)

            subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_canopyHeight_cube.time.values}, axis=list(subset_canopyHeight_cube.dims).index('time'))
            subset_agb_cube    = self.agb_cube.sel(IN).astype('float16').sel(time = self.upscaling_config['output_writer_params']['dims']['time'])

            agb_var = self.DataConfig['agb_var_cube']
            agb_feature_name = [x for x in self.DataConfig['features'] if "agb" in x][0]
            mean_agb_da = subset_agb_cube[agb_var]
            mean_agb_vals = mean_agb_da.values
            std_agb_vals = subset_agb_cube[agb_var + '_std'].values
            member_rngs = self._member_rngs(IN)
            self._ensure_extrapolation_index()

            mask_intact_forest = ~np.zeros((subset_canopyHeight_cube.sizes['latitude'], subset_canopyHeight_cube.sizes['longitude']), dtype=bool)
            for _, row in self.intact_tropical_forest.iterrows():
                polygon = row.geometry
                polygon_mask = geometry_mask([polygon], out_shape=mask_intact_forest.shape, 
                                             transform=subset_canopyHeight_cube.rio.transform())
                
                if False in polygon_mask:
                    mask_intact_forest[polygon_mask==False] = False
            mask_intact_forest = mask_intact_forest.reshape(-1)
            
            ML_pred_age_members = {year: [] for year in all_years}
            ML_pred_class_proba_members = {year: [] for year in all_years}
            ML_extrapolation_index_members = {year: [] for year in all_years}
            for run_ in np.arange(self.upscaling_config['num_members']):

                noise = member_rngs[run_].standard_normal(mean_agb_vals.shape)
                perturbed_agb_vals = np.clip(mean_agb_vals + std_agb_vals * noise, 0, None).astype('float16')
                subset_agb_cube_member = xr.DataArray(
                    perturbed_agb_vals, coords=mean_agb_da.coords, dims=mean_agb_da.dims
                ).to_dataset(name=agb_feature_name)

                subset_features_cube     = xr.merge([subset_agb_cube_member, subset_clim_cube, subset_canopyHeight_cube])
                year_of_time_index = [pd.Timestamp(t).year for t in subset_features_cube.time.values]

                with open(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = "Classifier", id_ = run_), 'rb') as f:
                    classifier_config = pickle.load(f)
                best_classifier = classifier_config['best_model']
                features_classifier = classifier_config['selected_features']
                norm_stats_classifier = classifier_config['norm_stats']
                
                with open(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = 'Regressor', id_ = run_), 'rb') as f:
                    regressor_config = pickle.load(f)
                best_regressor = regressor_config['best_model']
                features_regressor = regressor_config['selected_features']
                norm_stats_regressor = regressor_config['norm_stats']
                
                all_features = list(np.unique(features_classifier + features_regressor))
                                   
                X_upscale = []
                for var_name in all_features:
                    if self.algorithm == "MLP":
                        X_upscale.append(self.norm(subset_features_cube[var_name], norm_stats_classifier[var_name]))
                    else:
                        X_upscale.append(subset_features_cube[var_name])
                    
                X_upscale_flattened = []
        
                for arr in X_upscale:
                    data = arr.data.flatten()
                    X_upscale_flattened.append(data)
                    
                X_upscale_flattened = da.array(X_upscale_flattened).transpose().compute()

                X_ei = da.array(
                    [subset_features_cube[v].data.flatten() for v in self.extrapolation_index_features]
                ).transpose().compute()

                ei_scores = self.extrapolation_index.score(X_ei).reshape(
                    len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1
                )

                ML_pred_class = np.zeros(X_upscale_flattened.shape[0]) * np.nan
                ML_pred_class_proba = np.zeros(X_upscale_flattened.shape[0]) * np.nan
                ML_pred_age = np.zeros(X_upscale_flattened.shape[0]) * np.nan

                mask = (np.all(np.isfinite(X_upscale_flattened), axis=1))

                if (X_upscale_flattened[mask].shape[0]>0):
                    index_mapping_class = [all_features.index(feature) for feature in features_classifier]
                    index_mapping_reg = [all_features.index(feature) for feature in features_regressor]

                    if self.algorithm == "XGBoost":
                        dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_class])
                        pred_class_proba = best_classifier.predict(dpred)
                        pred_class = (pred_class_proba > 0.5).astype('int16')

                    elif self.algorithm == "MLP":
                        dpred =  X_upscale_flattened[mask][:, index_mapping_class]
                        dpred = np.stack([self.norm(dpred[:, features_classifier.index(var_name)], norm_stats_classifier[var_name]) for var_name in features_classifier], axis=1)
                        pred_class = best_classifier.predict(dpred)
                        if hasattr(best_classifier, "predict_proba"):
                            pred_class_proba = best_classifier.predict_proba(dpred)[:, 1]
                        else:
                            pred_class_proba = pred_class.astype('float32')

                    elif self.algorithm == "AutoML":
                        dpred =  X_upscale_flattened[mask][:, index_mapping_class]
                        dpred = pd.DataFrame(dpred, columns = features_classifier)
                        pred_class = best_classifier.predict(dpred).values
                        if hasattr(best_classifier, "predict_proba"):
                            pred_class_proba = best_classifier.predict_proba(dpred).values[:, 1]
                        else:
                            pred_class_proba = pred_class.astype('float32')

                    else:
                        dpred =  X_upscale_flattened[mask][:, index_mapping_class]
                        pred_class = best_classifier.predict(dpred)
                        if hasattr(best_classifier, "predict_proba"):
                            pred_class_proba = best_classifier.predict_proba(dpred)[:, 1]
                        else:
                            pred_class_proba = pred_class.astype('float32')

                    ML_pred_class[mask] = pred_class
                    ML_pred_class_proba[mask] = pred_class_proba

                    if self.algorithm == "XGBoost":
                        dpred =  xgb.DMatrix(X_upscale_flattened[mask][:, index_mapping_reg])
                        pred_reg= best_regressor.predict(dpred)
                        
                    elif self.algorithm == "MLP":
                        dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
                        dpred = np.stack([self.norm(dpred[:, features_regressor.index(var_name)], norm_stats_regressor[var_name]) for var_name in features_regressor], axis=1)
                        pred_reg= best_regressor.predict(dpred)
                        
                    else:
                        dpred =  X_upscale_flattened[mask][:, index_mapping_reg]
                        pred_reg= best_regressor.predict(dpred)
                    
                    pred_reg[pred_reg>=self.DataConfig['max_forest_age'][0]] = self.DataConfig['max_forest_age'][0] -1
                    pred_reg[pred_reg<1] = 1
                    ML_pred_age[mask] = np.round(pred_reg).astype("int16")
                    ML_pred_age[ML_pred_class==1] = self.DataConfig['max_forest_age'][0]
                    ML_pred_age   = ML_pred_age.reshape(len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1)
                    ML_pred_class_proba_reshaped = ML_pred_class_proba.reshape(
                        len(subset_features_cube.latitude), len(subset_features_cube.longitude), len(subset_features_cube.time), 1
                    )

                    for i, year in enumerate(year_of_time_index):
                        age_i = ML_pred_age[:, :, i, :].reshape(-1)
                        if year == anchor_year:
                            age_i[~mask_intact_forest] = self.DataConfig['max_forest_age'][0]
                        ML_pred_age_members[year].append(age_i)
                        ML_pred_class_proba_members[year].append(ML_pred_class_proba_reshaped[:, :, i, :].reshape(-1))
                        ML_extrapolation_index_members[year].append(ei_scores[:, :, i, :].reshape(-1))

                else:
                    # nothing to predict here -> just NaN arrays with correct shape
                    n_px = len(subset_features_cube.latitude) * len(subset_features_cube.longitude)
                    for year in year_of_time_index:
                        ML_pred_age_members[year].append(np.full(n_px, np.nan, dtype="float32"))
                        ML_pred_class_proba_members[year].append(np.full(n_px, np.nan, dtype="float32"))
                        ML_extrapolation_index_members[year].append(np.full(n_px, np.nan, dtype="float32"))

            ML_pred_age_members = {y: np.stack(v, axis=0) for y, v in ML_pred_age_members.items()}
            ML_pred_class_proba_members = {y: np.stack(v, axis=0) for y, v in ML_pred_class_proba_members.items()}
            ML_extrapolation_index_members = {y: np.stack(v, axis=0) for y, v in ML_extrapolation_index_members.items()}
            sigma_ml_by_year = {y: np.nanstd(v, axis=0) for y, v in ML_pred_age_members.items()}

            sigma_B_meas_by_year = {}
            mean_B_meas_by_year = {}
            for year in all_years:
                year_ts = pd.Timestamp(f"{year}-01-01")
                sigma_B_meas_by_year[year] = subset_agb_cube[self.DataConfig['agb_var_cube'] + '_std'].sel(time=year_ts).values.reshape(-1)
                mean_B_meas_by_year[year]  = subset_agb_cube[self.DataConfig['agb_var_cube']].sel(time=year_ts).values.reshape(-1)

            # --- CR-based bias correction (static across years/members) ---
            cr_params = {
                "A": subset_CRparams_cube.A.values.reshape(-1),
                "b": subset_CRparams_cube.b.values.reshape(-1),
                "k": subset_CRparams_cube.k.values.reshape(-1)
            }
            cr_errors = {
                "A": subset_CRparams_cube.A_error.values.reshape(-1),
                "b": subset_CRparams_cube.b_error.values.reshape(-1),
                "k": subset_CRparams_cube.k_error.values.reshape(-1)
            }
            n_px = len(LTSD_anchor_flat)
            TSD = np.repeat(1, n_px)
            tmax = np.repeat(301, n_px)

            for run_ in range(self.upscaling_config['num_members']):
                ML_pred_age_by_year = {y: ML_pred_age_members[y][run_, :] for y in all_years}
                ML_pred_class_proba_by_year = {y: ML_pred_class_proba_members[y][run_, :] for y in all_years}
                ML_extrapolation_index_by_year = {y: ML_extrapolation_index_members[y][run_, :] for y in all_years}

                fusion_config = dict(self.DataConfig.get("age_fusion", {}))
                fusion_config.update({
                    "start_year": int(self.DataConfig['start_year'].split('-')[0]),
                    "end_year": int(self.DataConfig['end_year'].split('-')[0]),
                })
                fusion = AgeFusion(config=fusion_config)

                (age_by_year, provenance_by_year, extrapolation_by_year,
                 replacement_probability_by_year) = fusion.fuse_multi_year(
                    anchor_year=anchor_year,
                    ML_pred_age_by_year=ML_pred_age_by_year,
                    LTSD_by_year=LTSD_by_year,
                    biomass_by_year=mean_B_meas_by_year,
                    biomass_std_by_year=sigma_B_meas_by_year,
                    ml_std_by_year=sigma_ml_by_year,
                    cr_params=cr_params, cr_errors=cr_errors,
                    TSD=TSD, tmax=tmax,
                    ml_old_growth_proba_by_year=ML_pred_class_proba_by_year,
                    ml_extrapolation_index_by_year=ML_extrapolation_index_by_year,
                    return_replacement_probability=True,
                )

                out_shape = (len(subset_features_cube.latitude), len(subset_features_cube.longitude), 1, 1)

                def _to_da(arr, year, dtype):
                    return xr.DataArray(
                        arr.reshape(out_shape).astype(dtype),
                        coords={"latitude": subset_features_cube.latitude,
                                "longitude": subset_features_cube.longitude,
                                "time": [pd.Timestamp(f"{year}-01-01")],
                                'members': [run_]},
                        dims=["latitude", "longitude", "time", "members"])

                ds_by_year = []
                for year in all_years:
                    data_vars = {
                        "forest_age": _to_da(age_by_year[year], year, "float32"),
                        "forest_age_provenance": _to_da(provenance_by_year[year], year, "int8"),
                        "forest_age_extrapolation": _to_da(extrapolation_by_year[year], year, "float32"),
                    }
                    # Backward-compatible opt-in: existing cube configs need
                    # no change, while new products can expose the key
                    # disturbance diagnostic by declaring this variable.
                    if "stand_replacement_probability" in self.upscaling_config["cube_variables"]:
                        data_vars["stand_replacement_probability"] = _to_da(
                            replacement_probability_by_year[year], year, "float32"
                        )
                    ds_by_year.append(xr.Dataset(data_vars))

                # Concatenate across years and append the model member
                ds = xr.concat(ds_by_year, dim='time').transpose('members', 'latitude', 'longitude', 'time')

                self.pred_cube.CubeWriter(ds, n_workers=1)
                
    def model_tuning(self,
                     run_: int=1,
                     task_:str='Regressor',
                     feature_selection:bool=True,
                     feature_selection_method:str = 'recursive',                     
                     train_subset:dict ={},
                     valid_subset:dict ={}) -> None:
        """Perform model tuning using cross-validation.

        Parameters
        ----------
        run_ : int, optional
            Number of model runs (default is 1).
        task_ : str, optional
            The type of model to use for training (default is 'Regressor').
        feature_selection : bool, optional
            Whether to perform feature selection (default is True).
        feature_selection_method : str, optional
            The method for feature selection (default is 'recursive').
        train_subset : dict, optional
            Training subset for the model.
        valid_subset : dict, optional
            Validation subset for the model.
        
        Returns
        -------
        None
        """
        if feature_selection:
            self.DataConfig['features_selected'] = FeatureSelection(method=task_, 
                                                                   feature_selection_method = feature_selection_method, 
                                                                   features = self.DataConfig['features'],
                                                                   data = xr.open_dataset(self.DataConfig['training_dataset'])).get_features(n_jobs = self.n_jobs)
                
        else: 
            self.DataConfig['features_selected'] = self.DataConfig['features'].copy()
            
        if self.algorithm == "MLP":
            
            ml_method = MLPmethod(study_dir=self.study_dir, 
                                   DataConfig= self.DataConfig,
                                   method=self.algorithm + task_)
        elif self.algorithm == "XGBoost":
            ml_method = XGBoost(study_dir=self.study_dir, 
                                DataConfig= self.DataConfig,
                                method=self.algorithm + task_)
        elif self.algorithm == "RandomForest":
            ml_method = RandomForest(study_dir=self.study_dir, 
                                     DataConfig= self.DataConfig,
                                     method=self.algorithm + task_)
        
        ml_method.train(train_subset=train_subset,
                        valid_subset=valid_subset,
                        n_jobs = self.n_jobs)
        
        if not os.path.exists(self.study_dir + '/save_model/'):
             os.makedirs(self.study_dir + '/save_model/')
             
        with open(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = task_, id_ = run_), "wb") as fout:
            
            pickle.dump({'best_model': ml_method.best_model, 
                         'selected_features': self.DataConfig['features_selected'], 
                         'norm_stats' : ml_method.mldata.norm_stats}, fout)
    
    def model_training(self) -> None:
        """Train the model and generate high resolution maps of age using the trained model.
        """
        
        training_ds = xr.open_dataset(self.DataConfig['training_dataset'])
        cluster_ = training_ds.cluster.values

        # Weak-labelled clusters (dev/augment_og300_weaklabels.py) are kept
        # out of the train/valid split: they always go to training, so the
        # hyperparameter objective is scored on field-reference clusters
        # only, and the split of the real clusters is identical with or
        # without augmentation (given split_random_seed). That makes a
        # with/without weak-label ablation a clean single-factor comparison.
        if 'weak_label_source' in training_ds:
            is_weak = (training_ds['weak_label_source'].astype(str) != '').any('sample').values
        else:
            is_weak = np.zeros(len(cluster_), dtype=bool)

        train_subset, valid_subset = train_test_split(cluster_[~is_weak],
                                                      test_size=self.DataConfig['valid_fraction'],
                                                      shuffle=True,
                                                      random_state=self.DataConfig.get('split_random_seed'))
        train_subset = np.concatenate([train_subset, cluster_[is_weak]])
        print(f"Training clusters: {len(train_subset)} ({is_weak.sum()} weak-labelled), "
              f"validation clusters: {len(valid_subset)}")
        
        for run_ in tqdm(np.arange(self.upscaling_config['num_members']), desc='Training model members'):
            
            for task_ in ["Regressor", "Classifier"]:
                if not os.path.exists(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = task_, id_ = run_)):
                    self.model_tuning(run_ = run_, 
                                      task_ = task_,
                                      feature_selection= self.DataConfig['feature_selection'],
                                      feature_selection_method = self.DataConfig['feature_selection_method'],     
                                      train_subset=train_subset, 
                                      valid_subset=valid_subset)
        
        if os.path.exists(os.path.join(self.study_dir, "tune")):
            shutil.rmtree(os.path.join(self.study_dir, "tune"))
            
        if os.path.exists(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/cube_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/cube_sync_{self.task_id}.zarrsync"))
        
    def ForwardRun(self, 
                   task_id=None) -> None:
        """Perform the forward run of the model, generating high resolution predictions.
        
        Parameters
        ----------
        task_id : int, optional
            Task ID for parallel processing (default is None).
        """
                                
        lat_chunk_size, lon_chunk_size = self.pred_cube.cube.chunks['latitude'][0], self.pred_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.pred_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.pred_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
            
            lat_values = self.pred_cube.cube.latitude.values[lat_slice]
            lon_values = self.pred_cube.cube.longitude.values[lon_slice]

            # Select the extent based on the slice indices
            selected_extent = {"latitude": slice(lat_values[0], lat_values[-1]), 
                               "longitude": slice(lon_values[0], lon_values[-1])}
            
            # Process the chunk
            self.process_chunk(selected_extent)
        
        else:
           print(f"Task ID {task_id} is out of range. No chunk to process.")

        if os.path.exists(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync"))
        
    def process_chunk(self, extent) -> None:
        """Process a data chunk for prediction.
       
       Parameters
       ----------
       extent : dict
           Dictionary defining the spatial extent to process.
       """
        
        self._predict_func(extent).compute()
                
    def ParallelResampling(self, 
                           n_jobs:int=20) -> None:
        """Perform parallel resampling of the data.
        
        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs (default is 20).
        """
        
        member_out = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit a future for each member
            futures = [executor.submit(self.AgeResample, member_) 
                       for member_ in np.arange(self.upscaling_config['num_members'])]
            
            # As each future completes, get the result and add it to member_out
            for future in concurrent.futures.as_completed(futures):
                try:
                    member_out.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")

        xr.concat(member_out, dim = 'members').to_zarr(self.upscaling_config['AgeResample_cube'] + '_{resolution}deg'.format(resolution = str(self.upscaling_config['resample_resolution'])), mode= 'w')
        
    def AgeResample(self, member_:int=0) -> xr.Dataset:
        """
        Calculate the age fraction and resample the data.

        This function processes forest age class data using the Global Age Mapping Integration dataset,
        calculating the age fraction distribution changes over time. The results are saved as raster files.
        
        Attributes
        ----------
        member_ : int, optional
            Model member index (default is 0).
        """
           
        age_cube = xr.open_zarr(self.upscaling_config['cube_location']).sel(members = member_).drop_vars('members')
        zarr_out_ = []
        
        for var_ in set(age_cube.variables.keys()) - set(age_cube.dims):
            
            LatChunks = np.array_split(age_cube.latitude.values, self.upscaling_config['n_chunks'])
            LonChunks = np.array_split(age_cube.longitude.values, self.upscaling_config['n_chunks'])
            chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
        		        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
        		    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] 
            
            ds_ = []
            for year_ in age_cube.time.values:
            
                iter_ = 0
                for chunck in chunk_dict:
                    
                    chunck.update({'time': year_})
                    
                    data_chunk = age_cube[var_].sel(chunck).transpose('latitude', 'longitude')
                    data_chunk = data_chunk.where(np.isfinite(data_chunk), -9999).astype('float32')  
                    
                    data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                    data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                    data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
                    data_chunk.attrs = {'long_name': 'Forest age',
                                        'units': 'Ton /ha',
                                        'valid_max': 300,
                                        'valid_min': 0}
                    data_chunk.attrs["_FillValue"] = -9999  
                    out_dir = '{tmp_folder}/{member}/{var_}/'.format(tmp_folder = self.tmp_folder, member= str(member_), var_ = var_)
                    if not os.path.exists(out_dir):
               		    os.makedirs(out_dir)
                           
                    data_chunk.rio.to_raster(raster_path= out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)), 
                                             driver="COG", BIGTIFF='YES', compress=None,  dtype= 'float32')      
                    
                    gdalwarp_command = [
                                        'gdal_translate',
                                        '-a_nodata', '-9999',
                                        out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)),
                                        out_dir + '{var_}_{iter_}_nodata.tif'.format(var_ = var_, iter_=str(iter_))                
                                    ]
                    subprocess.run(gdalwarp_command, check=True)
                    os.remove(out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)))
                    
                    iter_ += 1
            
                input_files = glob.glob(os.path.join(out_dir, '*_nodata.tif'))
                vrt_filename = out_dir + '/{var_}.vrt'.format(var_ = var_)
                    
                gdalbuildvrt_command = [
                    'gdalbuildvrt',
                    vrt_filename
                ] + input_files
                    
                subprocess.run(gdalbuildvrt_command, check=True)
                    
                gdalwarp_command = [
                    'gdalwarp',
                    '-srcnodata', '-9999',
                    '-dstnodata', '-9999',
                    '-tr', str(self.upscaling_config['resample_resolution']), str(self.upscaling_config['resample_resolution']),
                    '-t_srs', 'EPSG:4326',
                    '-of', 'Gtiff',
                    '-te', '-180', '-90', '180', '90',
                    '-r', 'average',
                    '-ot', 'Float32',
                    '-co', 'COMPRESS=LZW',
                    '-co', 'BIGTIFF=YES',
                    '-overwrite',
                    f'/{vrt_filename}',
                    out_dir + f'/{var_}_{self.upscaling_config["resample_resolution"]}deg_{year_}.tif'.format(var_=var_, year_= str(year_)),
                ]
                subprocess.run(gdalwarp_command, check=True)
                
                for tif_file in input_files:
                      os.remove(tif_file)
                
                da_ =  rio.open_rasterio(out_dir + f'/{var_}_{self.upscaling_config["resample_resolution"]}deg_{year_}.tif'.format(var_=var_, year_= str(year_)))
                da_ =  da_.rename({"x": 'longitude', "y": 'latitude', 'band': "time"}).to_dataset(name = var_)
                da_['time'] = [year_]
                
                ds_.append(da_)
            zarr_out_.append(xr.concat(ds_, dim='time'))
            
        return xr.merge(zarr_out_).expand_dims({"members": [member_]})

                        
    def norm(self, 
             x: np.array,
             norm_stats:dict) -> np.array:
        """Normalize an array of values using the given normalization statistics.

        Parameters
        ----------
        x : np.array
            The array of values to normalize.
        norm_stats : dict
            A dictionary containing the normalization statistics, with keys 'mean' and 'std'.
    
        Returns
        -------
        np.array
            The normalized array of values.
        """
        
        return (x - norm_stats['mean']) / norm_stats['std']
