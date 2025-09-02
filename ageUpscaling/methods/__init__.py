# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de

from ageUpscaling.methods.feature_selection import FeatureSelection
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.methods.xgboost import XGBoost
from ageUpscaling.methods.RandomForest import RandomForest
from ageUpscaling.methods.CRBayesAgeFuser import CRBayesAgeFuser
from ageUpscaling.methods.AgeFusion import AgeFusion

__all__ = [
    'FeatureSelection',
    'MLPmethod',
    'XGBoost',
    'RandomForest',
    'AgeFusion',
    'CRBayesAgeFuser'
]
