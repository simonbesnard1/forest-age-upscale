#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   biomass_uncertainty.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam

This module defines a method class for handling the creation and updating of biomass uncertainty data cubes.

Example usage:
--------------
from biomass_uncertainty import BiomassUncertainty

# Create a BiomassUncertainty instance
config_path = 'path/to/config.yml'
study_dir = 'path/to/study_dir'

biomass_uncertainty = BiomassUncertainty(Config_path=config_path, study_dir=study_dir)
biomass_uncertainty.BiomassUncertaintyCubeInit()
biomass_uncertainty.BiomassUncertaintyCalc(task_id=0)
"""
import os
import shutil
from itertools import product
from abc import ABC

import numpy as np
import yaml as yml

import dask

import xarray as xr
import zarr

from ageUpscaling.core.cube import DataCube

class BiomassUncertainty(ABC):
    """A class for managing the biomass uncertainty data cubes.

    Parameters
    ----------
    Config_path : str
        Path to the configuration file.
    study_dir : str, optional
        Directory for the study.
    n_jobs : int, optional
        Number of workers. Default is 1.
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
        self.agb_cube = xr.open_zarr(self.config_file['Biomass_cube'], synchronizer=self.sync_feature).sel(time = self.config_file['output_writer_params']['dims']['time'])
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/agbUnc_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.agb_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.agb_cube.longitude.values
        self.config_file['output_writer_params']['dims']['members'] =  self.config_file['n_members']
        
        self.agb_members_cube = DataCube(cube_config = self.config_file)
        
        np.random.seed(0)
        self.permutation_k = [np.clip(np.random.normal(0, 1) / (3 * 1), -1, 1) for i in np.arange(self.config_file['n_members'])]

    @dask.delayed
    def _calc_func(self, 
                   IN) -> None:
        """
        Calculate the biomass uncertainty based on the biomass mean and standard deviation from the config file.

        Parameters
        ----------
        IN : dict
            Dictionary for subsetting the agb_cube.
        """
        
        subset_agbMean_cube = self.agb_cube.isel(IN)['aboveground_biomass']
        subset_agbMean_cube = subset_agbMean_cube.where(subset_agbMean_cube>0)
        
        subset_agbStd_cube = self.agb_cube.isel(IN)['aboveground_biomass_std']
        subset_agbStd_cube = subset_agbStd_cube.where(subset_agbMean_cube>0)
        
        if not np.isnan(subset_agbMean_cube.isel(time=0)).all():
        
            generated_maps = []
            
            for time_step in self.agb_cube.time:
                time_maps = []
                mean_agb_at_time = subset_agbMean_cube.sel(time=time_step)
                std_agb_at_time = subset_agbStd_cube.sel(time=time_step)
                            
                for k in range(self.config_file['n_members']):
                    
                    S_k = self.permutation_k[k]
    
                    new_map_data = np.maximum(mean_agb_at_time + S_k * std_agb_at_time, 0)
                    
                    new_map_data = new_map_data.expand_dims({"members": [k]})
                    
                    time_maps.append(new_map_data)
                
                # Combine all maps for this time step into a single Dataset
                combined_maps = xr.concat(time_maps, dim = 'members')
                generated_maps.append(combined_maps)
    
            # Combine all time step datasets into a single xarray Dataset
            out_cube = xr.concat(generated_maps,dim='time').transpose("members", 'latitude', 'longitude', 'time').to_dataset(name='aboveground_biomass')
    
            self.agb_members_cube.CubeWriter(out_cube, n_workers=1)
             
    def BiomassUncertaintyCubeInit(self):
        """
        Initialize the biomass uncertainty cube.
        """
        
        self.agb_members_cube.init_variable(self.config_file['cube_variables'])
    
    def BiomassUncertaintyCalc(self,
                               task_id=None) -> None:
        """
        Calculate the biomass uncertainty for each chunk of data.
        """

        lat_chunk_size, lon_chunk_size = self.agb_members_cube.cube.chunks['latitude'][0], self.agb_members_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.agb_members_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.agb_members_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
     
            # Select the extent based on the slice indices
            selected_extent = {"latitude": lat_slice, "longitude": lon_slice}
            
            # Process the chunk
            self.process_chunk(selected_extent)
        
        else:
           print(f"Task ID {task_id} is out of range. No chunk to process.")
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbUnc_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbUnc_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbUnc_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbUnc_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        """
        Process a chunk of the biomass uncertainty cube.
        """        
        self._calc_func(extent).compute()
        

                
    