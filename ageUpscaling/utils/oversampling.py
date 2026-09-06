#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File    :   oversampling.py

Shared stratified-oversampling helpers used by the RandomForest, XGBoost and
MLP training routines.

Both the regressor and classifier used the same one-dimensional,
age-only oversampling (regressor) or no oversampling at all (classifier).
That leaves two related gaps:

- The classifier's positive ("old growth") class is trained almost
  exclusively on high-biomass examples (the reference data barely has any
  low-biomass old-growth plots), so it never learns that low biomass does
  not rule out old growth.
- The regressor's age-bin oversampling ignores biomass, so a given age bin
  can still be dominated by whichever biomass sub-population happens to be
  most common in the reference data.

`stratified_oversample_extra_indices` generalises the original single-key
oversampling to an arbitrary number of stratification keys (e.g. age bin
AND biomass bin jointly), so both gaps can be closed with the same
mechanism.
"""
from typing import Optional

import numpy as np


def quantile_bins(values: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Assign each value to one of up to `n_bins` quantile-based bins.

    Falls back to a single bin if there are too few distinct values to form
    `n_bins` non-degenerate quantile edges (e.g. a near-constant array).

    Parameters
    ----------
    values : np.ndarray
        1D array of values to bin (e.g. biomass).
    n_bins : int, default 4
        Target number of quantile bins.

    Returns
    -------
    np.ndarray
        Integer bin index per value.
    """
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.nanquantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, edges[1:-1], right=False)


def stratified_oversample_extra_indices(
    *strata: np.ndarray,
    max_multiple: Optional[float] = 10.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Compute indices of extra (bootstrap) samples to balance joint strata.

    Groups rows by the unique combination of the given stratification
    arrays (e.g. age bin, biomass bin), then draws with replacement from
    every non-empty combination so it reaches the size of the largest
    combination (capped by `max_multiple`, see below). Mirrors the original
    per-file oversampling logic (`np.random.choice(idx_, required_samples)`),
    generalised to more than one stratification key.

    Adding a second stratification key (e.g. biomass on top of age) makes
    small cells much smaller and much more numerous, so uncapped
    max-based balancing can end up bootstrapping a handful of real examples
    by several hundred to a thousand times over (checked empirically on the
    training data: some age x biomass cells hold a single example while the
    largest holds >1000, an uncapped duplication ratio >1000x). Duplicating
    a few rows that heavily is closer to memorising those specific points
    than teaching the model the underlying pattern. `max_multiple` bounds
    how many times any one stratum's real examples can be effectively
    repeated, at the cost of the rarest strata staying underrepresented
    relative to the largest.

    Parameters
    ----------
    *strata : np.ndarray
        One or more equal-length 1D arrays defining the joint
        stratification (e.g. `age_bin`, `biomass_bin`, or just a binary
        class label). Values are compared by equality, not numerically, so
        any hashable dtype works.
    max_multiple : float or None, default 10.0
        Cap the oversampling target for any stratum at `max_multiple`
        times its own original count (in addition to the overall max-count
        target). Pass `None` to disable the cap and match the original,
        uncapped behaviour.
    rng : np.random.RandomState, optional
        Random state to use for reproducibility. Defaults to the global
        `numpy.random` state.

    Returns
    -------
    np.ndarray
        Indices into the original (row-aligned) arrays, to be concatenated
        with the original data to produce the balanced set. Empty if no
        oversampling is needed.
    """
    if rng is None:
        rng = np.random

    strata = [np.asarray(s) for s in strata]
    n = len(strata[0])
    if n == 0:
        return np.array([], dtype=int)

    combined = strata[0].astype(str)
    for s in strata[1:]:
        combined = np.char.add(np.char.add(combined, "_"), s.astype(str))

    keys, counts = np.unique(combined, return_counts=True)
    global_target = counts.max()

    extra_idx = []
    for key, count in zip(keys, counts):
        target = global_target if max_multiple is None else min(global_target, count * max_multiple)
        required = int(target - count)
        if required > 0:
            idx = np.where(combined == key)[0]
            extra_idx.append(rng.choice(idx, required))

    if not extra_idx:
        return np.array([], dtype=int)
    return np.concatenate(extra_idx)
