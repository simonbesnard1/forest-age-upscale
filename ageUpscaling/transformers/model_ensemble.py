#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   extrapolation_index.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for calculating model ensemble
"""
import os
import shutil
from abc import ABC
from itertools import product
import atexit

import numpy as np
import yaml as yml

import dask
import xarray as xr
import zarr 

from ageUpscaling.core.cube import DataCube

synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class ModelEnsemble(ABC):
    """ModelEnsemble abstract class used for calculating model ensemble

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
                 cube_config_path: str=None,            
                 study_dir: str=None,
                 n_jobs: int = 1,
                 **kwargs):

        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
            
        self.study_dir = study_dir
        self.n_jobs = n_jobs
        self.cube_config['cube_location'] = os.path.join(self.study_dir, self.cube_config['cube_name'])
    
    def calculate_ensemble(self, 
                        IN) -> None:
        
        subset_age_cube  = xr.open_zarr(self.cube_config['cube_location'], synchronizer=synchronizer).sel(latitude= IN['latitude'],longitude=IN['longitude'])
        
        for var_ in {key for key in set(list(subset_age_cube.keys())) - set(subset_age_cube.coords) if not key.endswith('_median') and not key.endswith('_iqr')}:
            output_mean = subset_age_cube[var_].median(dim = 'members').to_dataset(name = var_ + '_median')
            output_iqr = subset_age_cube[var_].quantile(q=0.75, dim = 'members') - subset_age_cube[var_].quantile(q=0.25, dim = 'members') 
            output_iqr = output_iqr.to_dataset(name = var_ + '_iqr')
            output_ = xr.merge([output_mean, output_iqr])
            self.age_cube.update_cube(output_, initialize=False)
        
    def calculate_global_index(self) -> None:
        
        self.age_cube = DataCube(cube_config = self.cube_config)
        self.age_cube.init_variable(self.cube_config['cube_variables'], 
                                    njobs= len(self.cube_config['cube_variables'].keys()))            
        
        LatChunks = np.array_split(self.age_cube.cube.latitude.values, self.cube_config["num_chunks"])
        LonChunks = np.array_split(self.age_cube.cube.longitude.values, self.cube_config["num_chunks"])
        
        AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                       "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
   
        if (self.n_jobs > 1):
            with dask.config.set({'distributed.worker.memory.target': 50*1024*1024*1024, 
                                  'distributed.worker.threads': 2}):

                futures = [self.calculate_ensemble(i) for i in AllExtents]
                dask.compute(*futures, num_workers=self.n_jobs)    
        else:
            for extent in AllExtents:
                self.calculate_index(extent).compute()
        
            
