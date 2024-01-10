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

class DifferenceBiomass(ABC):
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
        sync_file_features = os.path.abspath(f"{study_dir}/features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        self.age_cube = xr.open_zarr(self.config_file['ForestAge_cube'], synchronizer=self.sync_feature)
        self.agb_cube = xr.open_zarr(self.config_file['Biomass_cube'], synchronizer=self.sync_feature)
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        
        self.agb_diff_cube = DataCube(cube_config = self.config_file)
        
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
        subset_agb_cube = self.agb_cube.sel(IN)[['aboveground_biomass']]
        subset_agb_cube = subset_agb_cube.where(subset_agb_cube>0)
        
        diff_age = subset_age_cube.sel(time= '2020-01-01') - subset_age_cube.sel(time= '2010-01-01')
        diff_agb = subset_agb_cube.sel(time= '2020-01-01') - subset_agb_cube.sel(time= '2010-01-01')
        
        stand_replaced_class = xr.where(diff_age < 0, 1, 0).where(np.isfinite(diff_age)).rename({self.config_file['forest_age_var']: 'stand_replaced_class'})
        growing_forest_class = xr.where(diff_age > 0, 1, 0).where(np.isfinite(diff_age)).rename({self.config_file['forest_age_var']: 'growing_forest_class'})
        stable_forest_class = xr.where(diff_age == 0, 1, 0).where(np.isfinite(diff_age)).rename({self.config_file['forest_age_var']: 'stable_forest_class'})
        diff_age = diff_age.rename({self.config_file['forest_age_var']: 'age_difference'})        
        diff_agb = diff_agb.rename({'aboveground_biomass': 'agb_difference'})        
        
        old_growth_2010 = subset_age_cube.sel(time= '2010-01-01').where(subset_age_cube.sel(time= '2010-01-01') > 150)
        very_young_2010 = subset_age_cube.sel(time= '2010-01-01').where(subset_age_cube.sel(time= '2010-01-01') < 21)
        intermediate_2010 = subset_age_cube.sel(time= '2010-01-01').where( (subset_age_cube.sel(time= '2010-01-01') > 20) & (subset_age_cube.sel(time= '2010-01-01') < 81) )
        mature_2010 = subset_age_cube.sel(time= '2010-01-01').where( (subset_age_cube.sel(time= '2010-01-01') > 80) & (subset_age_cube.sel(time= '2010-01-01') < 151) )
        
        old_growth_2010_replaced = old_growth_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        old_growth_diff_replaced = diff_age.where(np.isfinite(old_growth_2010_replaced)).rename({'age_difference': 'OG_stand_replaced_diff'})
        OG_stand_replaced_class = xr.where(old_growth_diff_replaced < 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'OG_stand_replaced_diff': 'OG_stand_replaced_class'})
        
        very_young_2010_replaced = very_young_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        very_young_diff_replaced = diff_age.where(np.isfinite(very_young_2010_replaced)).rename({'age_difference': 'Young_stand_replaced_diff'})
        very_young_stand_replaced_class = xr.where(very_young_diff_replaced < 10, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'Young_stand_replaced_diff': 'Young_stand_replaced_class'})
        
        intermediate_2010_replaced = intermediate_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        intermediate_diff_replaced = diff_age.where(np.isfinite(intermediate_2010_replaced)).rename({'age_difference': 'Intermediate_stand_replaced_diff'})
        intermediate_stand_replaced_class = xr.where(intermediate_diff_replaced < 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'Intermediate_stand_replaced_diff': 'Intermediate_stand_replaced_class'})
        
        mature_2010_replaced = mature_2010[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)        
        mature_diff_replaced = diff_age.where(np.isfinite(mature_2010_replaced)).rename({'age_difference': 'Mature_stand_replaced_diff'})
        mature_stand_replaced_class = xr.where(mature_diff_replaced < 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'Mature_stand_replaced_diff': 'Mature_stand_replaced_class'})
        
        old_growth_2010_aging = old_growth_2010[self.config_file['forest_age_var']].where(growing_forest_class.growing_forest_class==1)        
        old_growth_diff_aging = diff_age.where(np.isfinite(old_growth_2010_aging)).rename({'age_difference': 'OG_aging_diff'})
        OG_aging_class = xr.where(old_growth_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'OG_aging_diff': 'OG_aging_class'})
        
        very_young_2010_aging = very_young_2010[self.config_file['forest_age_var']].where(growing_forest_class.growing_forest_class==1)        
        very_young_diff_aging = diff_age.where(np.isfinite(very_young_2010_aging)).rename({'age_difference': 'Young_aging_diff'})
        very_young_aging_class = xr.where(very_young_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'Young_aging_diff': 'Young_aging_class'})
        
        intermediate_2010_aging = intermediate_2010[self.config_file['forest_age_var']].where(growing_forest_class.growing_forest_class==1)        
        intermediate_diff_aging = diff_age.where(np.isfinite(intermediate_2010_aging)).rename({'age_difference': 'Intermediate_aging_diff'})
        intermediate_aging_class = xr.where(intermediate_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'Intermediate_aging_diff': 'Intermediate_aging_class'})
        
        mature_2010_aging = mature_2010[self.config_file['forest_age_var']].where(growing_forest_class.growing_forest_class==1)        
        mature_diff_aging = diff_age.where(np.isfinite(mature_2010_aging)).rename({'age_difference': 'Mature_aging_diff'})
        mature_aging_class = xr.where(mature_diff_aging > 0, 1, 0).where(np.isfinite(diff_age.age_difference)).rename({'Mature_aging_diff': 'Mature_aging_class'})
        
        old_growth_agb_diff_replaced = diff_agb.where(OG_stand_replaced_class.OG_stand_replaced_class ==1).rename({'agb_difference': 'OG_agb_diff_replaced'})
        very_young_agb_diff_replaced = diff_agb.where(very_young_stand_replaced_class.Young_stand_replaced_class ==1).rename({'agb_difference': 'Young_agb_diff_replaced'})
        intermediate_agb_diff_replaced = diff_agb.where(intermediate_stand_replaced_class.Intermediate_stand_replaced_class ==1).rename({'agb_difference': 'Intermediate_agb_diff_replaced'})
        mature_agb_diff_replaced = diff_agb.where(mature_stand_replaced_class.Mature_stand_replaced_class ==1).rename({'agb_difference': 'Mature_agb_diff_replaced'})
        old_growth_agb_diff_aging = diff_agb.where(OG_aging_class.OG_aging_class ==1).rename({'agb_difference': 'OG_agb_diff_aging'})
        very_young_agb_diff_aging = diff_agb.where(very_young_aging_class.Young_aging_class ==1).rename({'agb_difference': 'Young_agb_diff_aging'})
        intermediate_agb_diff_aging = diff_agb.where(intermediate_aging_class.Intermediate_aging_class ==1).rename({'agb_difference': 'Intermediate_agb_diff_aging'})
        mature_agb_diff_aging = diff_agb.where(mature_aging_class.Mature_aging_class ==1).rename({'agb_difference': 'Mature_agb_diff_aging'})
        stable_agb_diff = diff_agb.where(stable_forest_class.stable_forest_class ==1).rename({'agb_difference': 'Stable_agb_diff'})  
        
        out_cube = xr.merge([old_growth_agb_diff_replaced, very_young_agb_diff_replaced, intermediate_agb_diff_replaced, mature_agb_diff_replaced,
                             old_growth_agb_diff_aging, very_young_agb_diff_aging, intermediate_agb_diff_aging, mature_agb_diff_aging,
                             stable_agb_diff])
        
        self.agb_diff_cube.CubeWriter(out_cube, n_workers=1)
             
    def BiomassDiffCubeInit(self):
        
        self.agb_diff_cube.init_variable(self.config_file['cube_variables'], 
                                         njobs= len(self.config_file['cube_variables'].keys()))
    
    def BiomassDiffCalc(self,
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
                
    