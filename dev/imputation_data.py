#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:30:12 2023

@author: simon
"""
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import yaml as yml
import pandas as pd
import numpy as np

with open('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/experiments/cross_validation/data_config_xgboost.yaml', 'r') as f:
    cube_config =  yml.safe_load(f)
dat_ = pd.read_csv('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_subsetFIA_v4_WithworlClim.csv')
dat_['agb_gapfilled'] = dat_['agb']
dat_['agb_gapfilled'] = dat_.apply(lambda row: row['agb_cci'] if pd.isna(row['agb_gapfilled']) and row['sourceName'] == 'ForestPlotsNet' else row['agb_gapfilled'], axis=1)
dat_['age'][dat_['age']>300] = 300
dat_['canopy_height_gapfilled'] = dat_['canopy_height']

X = dat_[cube_config["features"] + ['age']].to_numpy()
mask_nan = np.all(np.isfinite(X), axis=1)
imputer = KNNImputer()
X_fill = imputer.fit_transform(X)


imp = IterativeImputer(max_iter=100, random_state=0)
imp.fit(X)            
X_fill =  imp.transform(X)

for var_ in np.arange(len(cube_config["features"])):
    
    dat_[cube_config["features"][var_]] = X_fill[:, var_]
dat_.to_csv('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_subsetFIA_v6.csv')
