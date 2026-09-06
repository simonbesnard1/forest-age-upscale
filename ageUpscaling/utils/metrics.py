"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-FileCopyrightText: 2024 Basil Kraft
# SPDX-License-Identifier: EUPL-1.2
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File: metrics.py

Metrics used by ageUpscaling.diagnostic.report: rmse, nrmse, modeling
efficiency. Only values present in both arrays are used.
"""

import numpy as np
import warnings


def rmse_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        se = np.power(mod-obs, 2)
        mse = np.nanmean(se, axis=-1)
        rmse = np.sqrt(mse)

        return rmse


def nrmse_gufunc(mod, obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        se = np.power(mod-obs, 2)
        mse = np.nanmean(se, axis=-1)
        rmse = np.sqrt(mse)
        nrmse = rmse / np.mean(obs)

        return nrmse


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
