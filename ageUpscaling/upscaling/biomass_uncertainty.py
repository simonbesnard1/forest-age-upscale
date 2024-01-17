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
import subprocess
import glob

import numpy as np
import yaml as yml
import pandas as pd

import dask

import xarray as xr
import zarr
import rioxarray as rio

from ageUpscaling.core.cube import DataCube

class BiomassUncertainty(ABC):
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
        sync_file_features = os.path.abspath(f"{study_dir}/agbUnc_features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        self.agb_cube = xr.open_zarr(self.config_file['Biomass_cube'], synchronizer=self.sync_feature)
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/agbUnc_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.agb_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.agb_cube.longitude.values
        
        self.agb_members_cube = DataCube(cube_config = self.config_file)
        
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
        
        subset_agbMean_cube = self.agb_cube.sel(IN)[['aboveground_biomass']]
        subset_agbMean_cube = subset_agbMean_cube.where(subset_agbMean_cube>0)
        
        subset_agbStd_cube = self.agb_cube.sel(IN)[['aboveground_biomass']]
        subset_agbStd_cube = subset_agbStd_cube.where(subset_agbMean_cube>0)
        
        generated_maps = []
        
        # Loop over each time step
        for time_step in self.agb_cube.time:
            time_maps = []
            mean_agb_at_time = subset_agbMean_cube.sel(time=time_step)
            std_agb_at_time = subset_agbStd_cube.sel(time=time_step)
        
            for k in range(self.config_file['num_maps']):
                # Generate a single random scaling factor S_k for this map
                S_k = np.random.normal(0, 1)
                
                # Apply S_k to each grid cell's sigma and add to its AGB
                new_map_data = mean_agb_at_time + S_k * std_agb_at_time
                
                # Create a new xarray DataArray with this data
                new_map = xr.DataArray(new_map_data, dims=('longitude', 'latiude', 'time'), 
                                       coords={'longitude': mean_agb_at_time.coords['longitude'], 'latitude': mean_agb_at_time.coords['latitude'], 'time': [time_step]})
                
                time_maps.append(new_map)
            
            # Combine all maps for this time step into a single Dataset
            combined_maps = xr.concat(time_maps, pd.Index(range(self.config_file['num_maps']), name='members'))
            generated_maps.append(combined_maps)
        
        # Combine all time step datasets into a single xarray Dataset
        out_cube = xr.concat(generated_maps,dim='time')

        self.agb_members_cube.CubeWriter(out_cube, n_workers=1)
             
    def BiomassUncertaintyCubeInit(self):
        
        self.agb_members_cube.init_variable(self.config_file['cube_variables'])
    
    def BiomassUncertaintyCalc(self,
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
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbUnc_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbUnc_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbUnc_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbUnc_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()
        

                
    