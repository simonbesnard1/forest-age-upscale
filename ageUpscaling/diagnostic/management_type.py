#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de
# File    :   management_type.py

This module defines a method class for handling the creation and updating of management type data cubes.

Example usage:
--------------
from management_type import ManagementType

# Create a ManagementType instance
config_path = 'path/to/config.yml'
study_dir = 'path/to/study_dir'

management_type = ManagementType(Config_path=config_path, study_dir=study_dir)
management_type.ManagementCubeInit()
management_type.ManagementCalc(task_id=0)
management_type.ManagementResample()
"""
import os
import shutil
from itertools import product
from abc import ABC
import subprocess
import glob

import numpy as np
import yaml as yml

import dask

import xarray as xr
import zarr
import rioxarray as rio

from ageUpscaling.core.cube import DataCube

class ManagementType(ABC):
    """A class for handling the creation and updating of management type data cubes.

    Parameters
    ----------
    Config_path : str
        Path to the configuration file.
    study_dir : str, optional
        Path to the study directory.
    n_jobs : int, default=1
        Number of workers.
    """
    def __init__(self,
                 Config_path: str,
                 study_dir: str = None,
                 n_jobs: int = 1,
                 **kwargs):
        """Initialize the ManagementType instance."""

        with open(Config_path, 'r') as f:
            self.config_file =  yml.safe_load(f)
        
        self.study_dir = study_dir
        self.n_jobs = n_jobs
        
        sync_file = os.path.abspath(study_dir + '/features_sync.zarrsync')
        
        if os.path.isdir(sync_file):
            shutil.rmtree(sync_file)
            
        self.task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
        sync_file_features = os.path.abspath(f"{study_dir}/management_features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        self.management_type_cube = xr.open_zarr(self.config_file['ForestManagement_cube'], synchronizer=self.sync_feature).isel(time=0).drop('time')
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/management_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.management_type_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.management_type_cube.longitude.values
                
        self.management_cube = DataCube(cube_config = self.config_file)
        self.tmp_folder = os.path.join(self.config_file['tmp_dir'], 'managementType/')

    @dask.delayed
    def _calc_func(self, 
                   IN) -> None:
        
        """
        Calculate the fraction of data for each management category based on the management_cube and the management classes from config_file.

        Args
        ----
        IN : dict
            Dictionary for subsetting the management_cube.

        Returns
        -------
        - None
        """
        
        subset_management_cube = self.management_type_cube.sel(IN)[[self.config_file['management_var']]]
        
        intact_forests = xr.where(subset_management_cube ==11, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'intact_forests'})
        naturally_regenerated = xr.where(subset_management_cube ==20, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'naturally_regenerated'})
        planted_forest = xr.where(subset_management_cube ==31, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'planted_forest'})
        plantation_forest = xr.where(subset_management_cube ==32, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'plantation_forest'})
        oil_palm = xr.where(subset_management_cube ==40, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'oil_palm'})
        agroforestry = xr.where(subset_management_cube ==53, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'agroforestry'})
            
        out_cube = xr.merge([intact_forests, naturally_regenerated, planted_forest, plantation_forest, oil_palm, agroforestry])        
        self.management_cube.CubeWriter(out_cube, n_workers=1)  

    def ManagementCubeInit(self) -> None:
        """
        Initialize the management type data cube.
        """
        
        self.management_cube.init_variable(self.config_file['cube_variables'])
    
    def ManagementCalc(self,
                     task_id=None) -> None:
        """
        Calculate the fraction of each management class.        
        """
        lat_chunk_size, lon_chunk_size = self.management_cube.cube.chunks['latitude'][0], self.management_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.management_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.management_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
            
            lat_values = self.management_cube.cube.latitude.values[lat_slice]
            lon_values = self.management_cube.cube.longitude.values[lon_slice]

            # Select the extent based on the slice indices
            selected_extent = {"latitude": slice(lat_values[0], lat_values[-1]), 
                               "longitude": slice(lon_values[0], lon_values[-1])}
            
            # Process the chunk
            self.process_chunk(selected_extent)
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/management_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/management_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/management_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/management_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent) -> None:
        """
        Process a chunk of data based on the given extent.
        """
        
        self._calc_func(extent).compute()

    def ManagementResample(self) -> None:
        """
        Resample the management type data cube.

        This function processes management class data, resampling it over a target resolution and saving the results as raster files.

        The function performs the following operations:
        - Reads management class data.
        - Loops through each variable and management class in the dataset.
        - Transforms and attributes data.
        - Splits the geographical area into chunks.
        - Processes each chunk.
        - Saves processed data as raster files.
        - Merges and converts the output into Zarr format.
        """
                        
        management_cube = xr.open_zarr(self.config_file['cube_location'])
        zarr_out_ = []
        for var_ in set(management_cube.variables.keys()) - set(management_cube.dims):
                            
            out_dir = '{tmp_folder}/{var_}/'.format(tmp_folder = self.tmp_folder, var_ = var_)
            if not os.path.exists(out_dir):
       		    os.makedirs(out_dir)
             
            data_class =management_cube[var_].transpose('latitude', 'longitude')
            data_class = data_class.where(np.isfinite(data_class), -9999).astype("int16")     
            data_class.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
            data_class.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
            data_class = data_class.rio.write_crs("epsg:4326", inplace=True)
            data_class.attrs = {'long_name': 'Management fraction',
                                'units': 'adimensional',
                                'valid_max': 1,
                                'valid_min': 0}
            
            LatChunks = np.array_split(data_class.latitude.values, self.config_file['n_chunks'])
            LonChunks = np.array_split(data_class.longitude.values, self.config_file['n_chunks'])
            chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
        		           "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                          for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
            
            iter_ = 0
            for chunck in chunk_dict:
                
                data_chunk = data_class.sel(chunck)
                data_chunk.attrs["_FillValue"] = -9999    
                data_chunk = data_chunk.astype('int16')
                data_chunk.rio.to_raster(raster_path=out_dir + 'chunk_{iter_}.tif'.format(iter_=str(iter_)), 
                                         driver="COG", BIGTIFF='YES', compress=None, dtype="int16")                            
               
                gdalwarp_command = [
                                    'gdal_translate',
                                    '-a_nodata', '-9999',
                                    out_dir + 'chunk_{iter_}.tif'.format(iter_=str(iter_)),
                                    out_dir + 'chunk_{iter_}_nodata.tif'.format(iter_=str(iter_))                                                
                                ]
                subprocess.run(gdalwarp_command, check=True)
                os.remove(out_dir + 'chunk_{iter_}.tif'.format(iter_=str(iter_)))
                
                iter_ += 1
                  
            input_files = glob.glob(os.path.join(out_dir, '*_nodata.tif'))
            vrt_filename = out_dir + '/{var_}.vrt'.format(var_ = var_)
                
            gdalbuildvrt_command = [
                'gdalbuildvrt',
                vrt_filename
            ] + input_files
                
            subprocess.run(gdalbuildvrt_command, check=True)
                
            gdalwarp_command = [
                'gdalwarp',
                '-srcnodata', '-9999',
                '-dstnodata', '-9999',
                '-tr', str(self.config_file['target_resolution']), str(self.config_file['target_resolution']),
                '-t_srs', 'EPSG:4326',
                '-of', 'Gtiff',
                '-te', '-180', '-90', '180', '90',
                '-r', 'average',
                '-ot', 'Float32',
                '-co', 'COMPRESS=LZW',
                '-co', 'BIGTIFF=YES',
                '-overwrite',
                f'/{vrt_filename}',
                out_dir + f'{var_}_{self.config_file["target_resolution"]}deg.tif'.format(var_= var_),
            ]
            subprocess.run(gdalwarp_command, check=True)
            for file_ in input_files:
                os.remove(file_)
            
            da_ =  rio.open_rasterio(out_dir + f'{var_}_{self.config_file["target_resolution"]}deg.tif'.format(var_= var_))     
            da_ =  da_.isel(band=0).drop_vars('band').rename({'x': 'longitude', 'y': 'latitude'}).to_dataset(name = var_)
            
            zarr_out_.append(da_.transpose('latitude', 'longitude'))
            
        xr.merge(zarr_out_).to_zarr(self.config_file['ForestManagementResample_cube'] + '_{resolution}deg'.format(resolution = str(self.config_file['target_resolution'])), mode= 'w')
        
        try:
            shutil.rmtree(self.tmp_folder)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
        
