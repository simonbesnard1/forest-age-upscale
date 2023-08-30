from ageUpscaling.methods.feature_selection import FeatureSelection
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.methods.xgboost import XGBoost
from ageUpscaling.methods.RandomForest import RandomForest
from ageUpscaling.methods.autoML import AutoML


__all__ = [
    'FeatureSelection',
    'MLPmethod',
    'XGBoost',
    'RandomForest',
    'AutoML'
]
