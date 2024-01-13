#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   upscaling.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for upscaling MLP model
"""
import os
import shutil
from tqdm import tqdm
from itertools import product
from abc import ABC

import numpy as np
import yaml as yml

import dask

import xarray as xr
import zarr

from ageUpscaling.core.cube import DataCube

class DifferenceAge(ABC):
    """Study abstract class used for cross validation, model training, prediction.

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
                 Config_path: str,
                 study_dir: str = None,
                 n_jobs: int = 1,
                 **kwargs):

        with open(Config_path, 'r') as f:
            self.config_file =  yml.safe_load(f)
        
        self.study_dir = study_dir
        self.n_jobs = n_jobs
        
        sync_file = os.path.abspath(study_dir + '/features_sync.zarrsync')
        
        if os.path.isdir(sync_file):
            shutil.rmtree(sync_file)
            
        self.task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
        sync_file_features = os.path.abspath(f"{study_dir}/ageDiff_features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        self.age_cube = xr.open_zarr(self.config_file['ForestAge_cube'], synchronizer=self.sync_feature)
     
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/ageDiff_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        
        self.age_diff_cube = DataCube(cube_config = self.config_file)
        
    @dask.delayed
    def _calc_func(self, 
                   IN) -> None:
        
        """
          Calculate the fraction of data for each age category based on the age_cube and the age_classes from config_file.
        
          Args:
          - IN: Dictionary for subsetting the age_cube.
        
          Returns:
          - age_class_fraction: xarray DataArray with fractions for each age class.
        """
        
        subset_age_cube = self.age_cube.sel(IN)[[self.config_file['forest_age_var']]]
     
        diff_age = subset_age_cube.sel(time= '2020-01-01') - subset_age_cube.sel(time= '2010-01-01')
        stand_replaced_age = diff_age.where(diff_age < 0).rename({self.config_file['forest_age_var']: 'stand_replaced_diff'})
        aging_forest_age = diff_age.where(diff_age >= 0).rename({self.config_file['forest_age_var']: 'aging_forest_diff'})
        stand_replaced_class = xr.where(diff_age < 0, 1, 0).where(np.isfinite(diff_age)).rename({self.config_file['forest_age_var']: 'stand_replaced_class'})
        aging_forest_class = xr.where(diff_age >= 0, 1, 0).where(np.isfinite(diff_age)).rename({self.config_file['forest_age_var']: 'aging_forest_class'})
        diff_age = diff_age.rename({self.config_file['forest_age_var']: 'age_difference'})            
        
        young_2010 = subset_age_cube.sel(time= '2010-01-01').where(subset_age_cube.sel(time= '2010-01-01') < 21)
        maturing_2010 = subset_age_cube.sel(time= '2010-01-01').where( (subset_age_cube.sel(time= '2010-01-01') > 20) & (subset_age_cube.sel(time= '2010-01-01') < 81) )
        mature_2010 = subset_age_cube.sel(time= '2010-01-01').where( (subset_age_cube.sel(time= '2010-01-01') > 80) & (subset_age_cube.sel(time= '2010-01-01') < 201) )
        old_growth_2010 = subset_age_cube.sel(time= '2010-01-01').where(subset_age_cube.sel(time= '2010-01-01') > 200)
        
        old_growth_2010_replaced = old_growth_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        old_growth_diff_replaced = diff_age.where(np.isfinite(old_growth_2010_replaced)).rename({'age_difference': 'OG_stand_replaced_diff'})
        OG_stand_replaced_class = xr.where(old_growth_diff_replaced < 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'OG_stand_replaced_diff': 'OG_stand_replaced_class'})
        
        young_2010_replaced = young_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        young_diff_replaced = diff_age.where(np.isfinite(young_2010_replaced)).rename({'age_difference': 'young_stand_replaced_diff'})
        young_stand_replaced_class = xr.where(young_diff_replaced < 10, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'young_stand_replaced_diff': 'young_stand_replaced_class'})
        
        maturing_2010_replaced = maturing_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        maturing_diff_replaced = diff_age.where(np.isfinite(maturing_2010_replaced)).rename({'age_difference': 'maturing_stand_replaced_diff'})
        maturing_stand_replaced_class = xr.where(maturing_diff_replaced < 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'maturing_stand_replaced_diff': 'maturing_stand_replaced_class'})
        
        mature_2010_replaced = mature_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        mature_diff_replaced = diff_age.where(np.isfinite(mature_2010_replaced)).rename({'age_difference': 'mature_stand_replaced_diff'})
        mature_stand_replaced_class = xr.where(mature_diff_replaced < 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'mature_stand_replaced_diff': 'mature_stand_replaced_class'})
        
        old_growth_2010_aging = old_growth_2010[self.config_file['forest_age_var']].where(aging_forest_class.aging_forest_class==1)        
        old_growth_diff_aging = diff_age.where(np.isfinite(old_growth_2010_aging)).rename({'age_difference': 'OG_aging_diff'})
        OG_aging_class = xr.where(old_growth_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'OG_aging_diff': 'OG_aging_class'})
        
        young_2010_aging = young_2010[self.config_file['forest_age_var']].where(aging_forest_class.aging_forest_class==1)        
        young_diff_aging = diff_age.where(np.isfinite(young_2010_aging)).rename({'age_difference': 'young_aging_diff'})
        young_aging_class = xr.where(young_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'young_aging_diff': 'young_aging_class'})
        
        maturing_2010_aging = maturing_2010[self.config_file['forest_age_var']].where(aging_forest_class.aging_forest_class==1)        
        maturing_diff_aging = diff_age.where(np.isfinite(maturing_2010_aging)).rename({'age_difference': 'maturing_aging_diff'})
        maturing_aging_class = xr.where(maturing_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'maturing_aging_diff': 'maturing_aging_class'})
        
        mature_2010_aging = mature_2010[self.config_file['forest_age_var']].where(aging_forest_class.aging_forest_class==1)        
        mature_diff_aging = diff_age.where(np.isfinite(mature_2010_aging)).rename({'age_difference': 'mature_aging_diff'})
        mature_aging_class = xr.where(mature_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'mature_aging_diff': 'mature_aging_class'})
        
        out_cube = xr.merge([diff_age, stand_replaced_age, aging_forest_age, stand_replaced_class, aging_forest_class,
                             old_growth_diff_replaced, OG_stand_replaced_class, young_diff_replaced, young_stand_replaced_class,
                             maturing_diff_replaced, maturing_stand_replaced_class, mature_diff_replaced, mature_stand_replaced_class,
                             old_growth_diff_aging, OG_aging_class, young_diff_aging, young_aging_class, maturing_diff_aging, 
                             maturing_aging_class, mature_diff_aging, mature_aging_class])
        
        self.age_diff_cube.CubeWriter(out_cube, n_workers=1)
             
    def AgeDiffCubeInit(self):
        
        self.age_diff_cube.init_variable(self.config_file['cube_variables'])
    
    def AgeDiffCalc(self,
                     task_id=None) -> None:
        """Calculate the fraction of each age class.
        
        """
        LatChunks = np.array_split(self.config_file['output_writer_params']['dims']['latitude'], self.config_file["num_chunks"])
        LonChunks = np.array_split(self.config_file['output_writer_params']['dims']['longitude'], self.config_file["num_chunks"])
        
        AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                       "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
        
        if  "SLURM_JOB_ID" in os.environ:
            selected_extent = AllExtents[task_id]
            
            self.process_chunk(selected_extent)
            
        else:
            if (self.n_jobs > 1):
                
                batch_size = 2
                for i in range(0, len(AllExtents), batch_size):
                    batch_futures = [self.process_chunk(extent) for extent in AllExtents[i:i+batch_size]]
                    dask.compute(*batch_futures, num_workers=self.n_jobs)
                            
            else:
                for extent in tqdm(AllExtents, desc='Calculating age class fraction'):
                    self.process_chunk(extent)
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()
                
    