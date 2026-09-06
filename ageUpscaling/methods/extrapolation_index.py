#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File    :   extrapolation_index.py

Pixel-based model extrapolation index, following the Area of Applicability
(AOA) methodology (Meyer & Pebesma, 2021, Methods in Ecology and Evolution):
a feature-importance-weighted nearest-neighbor distance in (standardized)
feature space, normalised by the typical nearest-neighbor distance within
the training data itself, so the result is an interpretable ratio rather
than an arbitrary raw distance -- DI ~= 1 means "as novel as a typical
training point", DI >> 1 means genuinely outside anything the model has
seen. `score()` clips DI/threshold to [0, 1] for a bounded index, where the
threshold is derived from the training data's own nearest-neighbor
distance distribution (a boxplot-style outlier cutoff), i.e. the Area of
Applicability boundary.

Feature weighting matters because raw (unweighted) distance treats every
feature as equally important -- being far from training data along an
unimportant dimension shouldn't count the same as being far along the
dimension the model actually relies on. Weights come from the trained
model's own SHAP feature importances (`compute_feature_weights`), not a
fresh throwaway model, so the index reflects what the deployed model
actually uses.
"""
import os
import shutil
from abc import ABC
from itertools import product
import atexit

import numpy as np
import yaml as yml

import dask
import dask.array as da
import xarray as xr
import zarr

from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import shap

from ageUpscaling.core.cube import DataCube


synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class ExtrapolationIndex(ABC):
    """Pixel-based model extrapolation (Area of Applicability) index.

    Parameters
    ----------
    DataConfig_path : DataConfig_path
        A data configuration path.
    out_dir : str
        The study base directory.
        See `directory structure` for further details.
    exp_name : str = 'exp_name'
        The experiment name.
        See `directory structure` for further details.
    study_dir : Optional[str] = None
        The restore directory. If passed, an existing study is loaded.
        See `directory structure` for further details.
    n_jobs : int = 1
        Number of workers.

    """
    def __init__(self,
                 DataConfig_path: str=None,
                 cube_config_path: str=None,
                 base_dir: str=None,
                 n_jobs: int = 1,
                 **kwargs):
        # Both config paths are optional: pass neither to use this class
        # purely as a `fit`/`score` extrapolation-index helper (e.g. from
        # UpscaleAge, to feed AgeFusion.fuse's ml_extrapolation_index_*
        # arguments) without the standalone calculate_global_index cube
        # workflow, which does require them.
        self.DataConfig = {}
        if DataConfig_path is not None:
            with open(DataConfig_path, 'r') as f:
                self.DataConfig = yml.safe_load(f)

        self.cube_config = {}
        if cube_config_path is not None:
            with open(cube_config_path, 'r') as f:
                self.cube_config = yml.safe_load(f)

        self.base_dir = base_dir
        self.n_jobs = n_jobs
        if cube_config_path is not None:
            self.cube_config['cube_location'] = os.path.join(self.base_dir, self.cube_config['cube_name'])

        self.scaler_ = None
        self.weights_ = None
        self.nn_ = None
        self.mean_train_distance_ = None
        self.threshold_ = None

    @staticmethod
    def compute_feature_weights(model, X, feature_names) -> dict:
        """Per-feature importance weights from an already-trained tree model.

        Uses SHAP's TreeExplainer on the model actually being deployed (not
        a separate throwaway model), so the extrapolation index reflects
        what that model relies on. Weights are mean(|SHAP value|) per
        feature, normalised to sum to 1.

        Parameters
        ----------
        model
            A trained tree-based model (e.g. an xgboost Booster) compatible
            with `shap.TreeExplainer`.
        X : array-like (n_samples, n_features)
            Feature matrix used to compute SHAP values (e.g. a sample of
            the training data), in the same column order as `feature_names`.
        feature_names : list[str]

        Returns
        -------
        dict[str, float]
            Feature name -> normalised importance weight.
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        importance = np.mean(np.abs(shap_values), axis=0)
        total = importance.sum()
        weights = importance / total if total > 0 else np.ones_like(importance) / len(importance)
        return dict(zip(feature_names, weights))

    def fit(self, X_train, feature_weights=None):
        """Fit the weighted nearest-neighbor structure and AOA threshold.

        Parameters
        ----------
        X_train : np.ndarray (n_samples, n_features)
            Training feature matrix (same feature order used later in
            `score`). Rows with any non-finite value are dropped.
        feature_weights : dict[str, float] or array-like, optional
            Per-feature importance weights (e.g. from
            `compute_feature_weights`), in the same column order as
            `X_train`. Defaults to uniform weights (unweighted distance)
            if not supplied.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=float)
        mask = np.all(np.isfinite(X_train), axis=1)
        X_train = X_train[mask]

        self.scaler_ = preprocessing.StandardScaler().fit(X_train)
        X_scaled = self.scaler_.transform(X_train)

        if feature_weights is None:
            self.weights_ = np.ones(X_train.shape[1])
        elif isinstance(feature_weights, dict):
            self.weights_ = np.array(list(feature_weights.values()), dtype=float)
        else:
            self.weights_ = np.asarray(feature_weights, dtype=float)
        X_weighted = X_scaled * np.sqrt(self.weights_)

        # n_neighbors=2: each training point's nearest *other* point (index
        # 0 is always the point itself, at distance 0).
        self.nn_ = NearestNeighbors(n_neighbors=2).fit(X_weighted)
        distances, _ = self.nn_.kneighbors(X_weighted)
        loo_distances = distances[:, 1]

        self.mean_train_distance_ = float(np.mean(loo_distances))
        normalized_loo_di = loo_distances / self.mean_train_distance_
        q75, q25 = np.percentile(normalized_loo_di, [75, 25])
        # Boxplot-style outlier cutoff on the training data's own DI
        # distribution -- the Area of Applicability boundary.
        self.threshold_ = float(q75 + 1.5 * (q75 - q25))

        return self

    def score(self, X_new) -> np.ndarray:
        """Extrapolation index in [0, 1] for new samples.

        0 = well within the training domain, 1 = at or beyond the Area of
        Applicability threshold. NaN where `X_new` has non-finite features.

        Parameters
        ----------
        X_new : np.ndarray (n_samples, n_features)
            Same feature order as the `X_train` passed to `fit`.
        """
        if self.nn_ is None:
            raise RuntimeError("ExtrapolationIndex.fit(...) must be called before score(...)")

        X_new = np.asarray(X_new, dtype=float)
        out = np.full(X_new.shape[0], np.nan, dtype=np.float32)
        mask = np.all(np.isfinite(X_new), axis=1)
        if not mask.any():
            return out

        X_scaled = self.scaler_.transform(X_new[mask])
        X_weighted = X_scaled * np.sqrt(self.weights_)
        distances, _ = self.nn_.kneighbors(X_weighted, n_neighbors=1)
        di = distances[:, 0] / self.mean_train_distance_

        out[mask] = np.clip(di / self.threshold_, 0.0, 1.0).astype(np.float32)
        return out

    def calculate_index(self,
                        IN) -> None:

        subset_agb_cube  = xr.open_zarr(self.DataConfig['agb_cube'], synchronizer=synchronizer).sel(latitude= IN['latitude'],longitude=IN['longitude'])
        subset_clim_cube = xr.open_zarr(self.DataConfig['clim_cube'], synchronizer=synchronizer).sel(latitude= IN['latitude'],longitude=IN['longitude'])[[x for x in self.DataConfig['features'] if "WorlClim" in x]]

        if not self.cube_config["high_res_pred"]:
            subset_agb_cube    = subset_agb_cube.rename({'agb_001deg_cc_min_{tree_cover}'.format(tree_cover = self.tree_cover) : 'agb'})

        subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_agb_cube.time.values}, axis=list(subset_agb_cube.dims).index('time'))
        subset_agb_cube  = subset_agb_cube.agb.where(subset_agb_cube.agb >0).to_dataset()
        subset_cube      = xr.merge([subset_agb_cube, subset_clim_cube])

        X_upscale_reg = []
        for var_name in self.DataConfig['features']:
            X_upscale_reg.append(subset_cube[var_name])

        X_upscale_reg_flattened = []

        for arr in X_upscale_reg:
            data = arr.data.flatten()
            X_upscale_reg_flattened.append(data)

        X_upscale_reg_flattened = da.array(X_upscale_reg_flattened).transpose().compute()

        EI_pred = self.score(X_upscale_reg_flattened)
        EI_pred = EI_pred.reshape(len(subset_cube.latitude), len(subset_cube.longitude), len(subset_cube.time))

        output_EI_xr = xr.DataArray(EI_pred,
                                    coords={"latitude": subset_cube.latitude,
                                            "longitude": subset_cube.longitude,
                                            "time": subset_cube.time},
                                    dims=["latitude", "longitude", "time"]).to_dataset(name="Extrapolation_Index")

        self.interpolation_cube.update_cube(output_EI_xr, initialize=False)

    def calculate_global_index(self) -> None:

        self.interpolation_cube = DataCube(cube_config = self.cube_config)
        self.interpolation_cube.init_variable(self.cube_config['cube_variables'],
                                              njobs= len(self.cube_config['cube_variables'].keys()))

        LatChunks = np.array_split(self.interpolation_cube.cube.latitude.values, self.cube_config["num_chunks"])
        LonChunks = np.array_split(self.interpolation_cube.cube.longitude.values, self.cube_config["num_chunks"])

        AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                       "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])}
                    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]

        self.x_train = xr.open_dataset(self.DataConfig['training_dataset'])[self.DataConfig['features']]
        self.y_train = xr.open_dataset(self.DataConfig['training_dataset'])[self.DataConfig['target']]

        X_train = self.x_train.to_array().transpose('cluster', 'sample', 'variable').values.reshape(-1, len(self.DataConfig['features']))
        Y_train = self.y_train.to_array().values.reshape(-1)
        mask_nan = np.all(np.isfinite(X_train), axis=1) & np.isfinite(Y_train)
        X_train, Y_train = X_train[mask_nan], Y_train[mask_nan]

        feature_weights = None
        reference_model_path = self.DataConfig.get('reference_model_path')
        if reference_model_path:
            import pickle
            with open(reference_model_path, 'rb') as f:
                reference_model = pickle.load(f)['best_model']
            feature_weights = self.compute_feature_weights(reference_model, X_train, self.DataConfig['features'])

        self.fit(X_train, feature_weights=feature_weights)

        for tree_cover in self.cube_config["tree_cover_tresholds"]:
            self.tree_cover = tree_cover
            if (self.n_jobs > 1):
                with dask.config.set({'distributed.worker.memory.target': 50*1024*1024*1024,
                                      'distributed.worker.threads': 2}):

                    futures = [self.calculate_index(i) for i in AllExtents]
                    dask.compute(*futures, num_workers=self.n_jobs)
            else:
                for extent in AllExtents:
                    self.calculate_index(extent).compute()
