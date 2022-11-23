#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:03:40 2022

@author: simon
"""
#%% Load library
from ageUpscaling.cube.cube import Cube
import xarray as xr
import warnings
from multiprocessing.pool import ThreadPool
import dask
threads = 3
dask.config.set(pool=ThreadPool(threads))
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

# %% Initiate precalculated cube for MCD43A
ds_ = xr.open_dataset("/home/simon/Documents/science/GFZ/projects/forest_age_upscale/data/training_data/training_data_ageMap_OG300_fullFIA.nc")
cluster_ = ds_.cluster.values
sample_ = ds_.sample.values
pred_cube = Cube('/home/simon/Documents/science/GFZ/projects/forest_age_upscale/output/model_output',
                 njobs=1,
                 coords={'cluster': cluster_,
                         'sample': sample_})
