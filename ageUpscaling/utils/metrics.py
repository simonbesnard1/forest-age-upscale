"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-FileCopyrightText: 2024 Basil Kraft
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File: metrics.py

Calculate metrics like correlation or rmse on multidimensional array along given dimentsions
using dask.

Metrics implemented:
* correlation           > xr_corr
* rmse                  > xr_rmse
* mean percentage error > xr_mpe
* bias                  > xr_bias
* modeling effficiency  > xr_mef

Only values present in both datasets are used to calculate metrics.

"""

import numpy as np
import xarray as xr
import warnings
from datetime import datetime
import bottleneck

def pearson_cor_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)
        valid_count = valid_values.sum(axis=-1)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        mod -= np.nanmean(mod, axis=-1, keepdims=True)
        obs -= np.nanmean(obs, axis=-1, keepdims=True)

        cov = np.nansum(mod * obs, axis=-1) / valid_count
        std_xy = (np.nanstd(mod, axis=-1) * np.nanstd(obs, axis=-1))

        corr = cov / std_xy

        return corr


def xr_corr(mod, obs, dim):
    m = xr.apply_ufunc(
        pearson_cor_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'corr', 'units': '-'})
    m.name = 'corr'
    return m


def covariance_gufunc(mod, obs):
    return ((mod - mod.mean(axis=-1, keepdims=True))
            * (obs - obs.mean(axis=-1, keepdims=True))).mean(axis=-1)

def pearson_correlation_gufunc(mod, obs):
    return covariance_gufunc(mod, obs) / (mod.std(axis=-1) * obs.std(axis=-1))

def spearman_correlation_gufunc(mod, obs):
    x_ranks = bottleneck.rankdata(mod, axis=-1)
    y_ranks = bottleneck.rankdata(obs, axis=-1)
    return pearson_correlation_gufunc(x_ranks, y_ranks)

def xr_spearman_corr(mod, obs, dim):
    m = xr.apply_ufunc(
        spearman_correlation_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
	output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'corr', 'units': '-'})
    m.name = 'corr'
    return m


def rmse_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        se = np.power(mod-obs, 2)
        mse = np.nanmean(se, axis=-1)
        rmse = np.sqrt(mse)

        return rmse


def xr_rmse(mod, obs, dim):
    m = xr.apply_ufunc(
        rmse_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'rmse'})
    m.name = 'rmse'
    return m

def nrmse_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        se = np.power(mod-obs, 2)
        mse = np.nanmean(se, axis=-1)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.mean(obs)

        return nrmse


def xr_nrmse(mod, obs, dim):
    m = xr.apply_ufunc(
        nrmse_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'rmse'})
    m.name = 'rmse'
    return m


def mpe_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mpe = 100 * np.nanmean((obs - mod) / obs, axis=-1)

        return mpe


def xr_mpe(mod, obs, dim):
    m = xr.apply_ufunc(
        mpe_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'mpe', 'units': '%'})
    m.name = 'mpe'
    return m


def bias_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return np.nanmean(mod, axis=-1) - np.nanmean(obs, axis=-1)


def xr_bias(mod, obs, dim):
    m = xr.apply_ufunc(
        bias_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'bias'})
    m.name = 'bias'
    return m


def varerr_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return np.square(mod.std(-1) - obs.std(-1))


def xr_varerr(mod, obs, dim):
    m = xr.apply_ufunc(
        varerr_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'varerr'})
    m.name = 'varerr'
    return m


def phaseerr_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return (1.0 - pearson_cor_gufunc(mod, obs)) * 2.0 * mod.std(-1) * obs.std(-1)


def xr_phaseerr(mod, obs, dim):
    m = xr.apply_ufunc(
        phaseerr_gufunc, mod, obs,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'phaseerr'})
    m.name = 'phaseerr'
    return m


def rel_bias_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(mod) & np.isfinite(obs)

        mod[~valid_values] = np.nan
        obs[~valid_values] = np.nan

        return (np.nanmean(mod, axis=-1) - np.nanmean(obs, axis=-1)) / np.nanmean(obs, axis=-1)


def xr_rel_bias(obs, mod, dim):
    m = xr.apply_ufunc(
        bias_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'relative bias'})
    m.name = 'rel_bias'
    return m


def mef_gufunc(x, y):
    # x is obs, y is mod
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        valid_values = np.isfinite(x) & np.isfinite(y)

        x[~valid_values] = np.nan
        y[~valid_values] = np.nan

        sse = np.nansum(np.power(x-y, 2), axis=-1)
        sso = np.nansum(
            np.power(y-np.nanmean(y, axis=-1, keepdims=True), 2), axis=-1)

        mef = 1.0 - sse / sso

        return mef


def xr_mef(obs, mod, dim):
    m = xr.apply_ufunc(
        mef_gufunc, obs, mod,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True)
    m.attrs.update({'long_name': 'mef', 'units': '-'})
    m.name = 'mef'
    return m


def get_metric(obs, mod, fun, dim='time', verbose=False):
    """Calculate a metric along a dimension.

    Metrics implemented:
    * correlation           > xr_corr
    * rmse                  > xr_rmse
    * mean percentage error > xr_mpe
    * bias                  > xr_bias
    * phaseerr              > xr_phaseerr
    * varerr                > xr_varerr
    * modeling effficiency  > xr_mef

    Only values present in both datasets are used to calculate metrics.

    Parameters
    ----------
    data: xarray.Dataset
        Dataset with data variables 'mod' (modelled) and 'obs' (observed).
    fun: Callable
        A function that takes three arguments: Modelled (xarray.DataArray), observed (xarray.DataArray)
        and the dimension along which the metric is calculated.
    dim: str
        The dimension name along which the metri is calculated, default is `time`.

    Returns
    ----------
    xarray.Dataset

    """

    return fun(obs, mod, dim)


def get_metrics(mod, obs, funs, dim='time', verbose=True):
    """Calculate multiple metrics along a dimension and combine into single dataset.

    Metrics implemented:      name
    * correlation           > xr_corr
    * rmse                  > xr_rmse
    * mean percentage error > xr_mpe
    * bias                  > xr_bias
    * phaseerr              > xr_phaseerr
    * varerr                > xr_varerr
    * modeling effficiency  > xr_mef

    Only values present in both datasets are used to calculate metrics.

    Parameters
    ----------
    mod: xarray.DataArray
        The modelled data.
    obs: xarray.DataArray
        The observed data.
    funs: Iterable[str]
        An iterable of function names (see `metrics implemented`).
    dim: str
        The dimension name along which the metri is calculated.
    verbose: bool
        Silent if False (True is default).

    Returns
    ----------
    xarray.Dataset

    """

    fun_lookup = {
        'corr': xr_corr,
        'rmse': xr_rmse,
        'mpe': xr_mpe,
        'bias': xr_bias,
        'rel_bias': xr_rel_bias,
        'mef': xr_mef,
        'varerr': xr_varerr,
        'phaseerr': xr_phaseerr
    }

    requested_str = ", ".join(funs)
    options_str = ", ".join(fun_lookup.keys())

    tic = datetime.now()

    if verbose:
        print(f'{timestr(datetime.now())}: calculating metrics [{requested_str}]')

    met_list = []
    for fun_str in funs:
        if verbose:
            print(f'{timestr(datetime.now())}: - {fun_str}')
        if fun_str not in fun_lookup:
            raise ValueError(
                f'Function `{fun_str}` not one of the implemented function: [{options_str}].'
            )
        fun = fun_lookup[fun_str]
        met_list.append(fun(mod, obs, dim).compute())

    met = xr.merge(met_list)

    toc = datetime.now()

    elapsed = toc - tic
    elapsed_mins = int(elapsed.seconds / 60)
    elapsed_secs = int(elapsed.seconds - 60 * elapsed_mins)

    if verbose:
        print(f'{timestr(datetime.now())}: done; elapsed time: {elapsed_mins} min {elapsed_secs} sec')

    return met


def timestr(t):
    return t.strftime("%m/%d/%Y, %H:%M:%S")


def _single_xr_quantile(x, q, dim):
    if isinstance(dim, str):
        dim = [dim]
    ndims = len(dim)
    axes = tuple(np.arange(ndims)-ndims)
    m = xr.apply_ufunc(
        np.nanquantile, x,
        input_core_dims=[dim],
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs=True,
        kwargs={'q': q, 'axis': axes})
    m.attrs.update({'long_name': f'{q}-quantile'})
    m.name = 'quantile'
    return m


def xr_quantile(x, q, dim):
    if not hasattr([1, 2], '__iter__'):
        q = [q]
    quantiles = []
    for i, q_ in enumerate(q):
        r = _single_xr_quantile(x, q_, dim).compute()
        quantiles.append(r)
    quantiles = xr.concat(quantiles, 'quantile')
    quantiles['quantile'] = q

    return quantiles
