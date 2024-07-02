#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   biomassDiff_partition.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam

A method class for biomass difference partitioning and resampling.

"""
import os
import shutil
from itertools import product
from abc import ABC
import subprocess
import glob
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml as yml

import dask

import xarray as xr
import zarr
import rioxarray as rio

from ageUpscaling.core.cube import DataCube

class BiomassDiffPartition(ABC):
    """A class used for calculating and resampling biomass difference partitioning.

    Parameters
    ----------
    Config_path : str
        Path to the configuration file.
    study_dir : str, optional
        Directory for storing study data. If not provided, a new study will be created.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is 1.
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
        
        sync_file_features = os.path.abspath(f"{study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        
        self.age_cube = xr.open_zarr(self.config_file['ForestAge_cube'], synchronizer=self.sync_feature)
        self.agb_cube = xr.open_zarr(self.config_file['Biomass_cube'], synchronizer=self.sync_feature)
        
        age_class = np.array(self.config_file['age_classes'])
        age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
        age_labels[-1] = '>' + age_labels[-1].split('-')[0]
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        self.config_file['output_writer_params']['dims']['age_class'] = age_labels
        self.config_file['output_writer_params']['dims']['members'] =  self.config_file['num_members']
        
        self.agbDiffPartition_cube = DataCube(cube_config = self.config_file)
        self.tmp_folder = os.path.join(self.config_file['tmp_dir'], 'biomassDiffPartition/')
        
    @dask.delayed
    def _calc_func(self, 
                   IN) -> None:
        
        """
        Calculate the fraction of data for each age category based on the age_cube and the age_classes from the config file.
        
        Parameters
        ----------
        IN : dict
            Dictionary for subsetting the age_cube.
        
        Returns
        -------
        None
        """
        
        for member_ in np.arange(self.config_file['num_members']):
        
            subset_age_cube = self.age_cube.sel(IN).sel(members=member_)[[self.config_file['forest_age_var']]]
            diff_age = subset_age_cube.sel(time= '2020-01-01') - subset_age_cube.sel(time= '2010-01-01')
            diff_age = diff_age.where(diff_age != 0, 10).where(np.isfinite(diff_age))
            
            if not np.isnan(diff_age.to_array().values).all():
    
                stand_replaced_class = xr.where(diff_age < 10, 1, 0).where(np.isfinite(diff_age)).rename({self.config_file['forest_age_var']: 'stand_replaced_class'})
                aging_forest_class = xr.where(diff_age >= 10, 1, 0).where(np.isfinite(diff_age)).rename({self.config_file['forest_age_var']: 'aging_forest_class'})
                diff_age = diff_age.rename({self.config_file['forest_age_var']: 'age_difference'})        
                
                age_class = np.array(self.config_file['age_classes'])
                age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
        
                for i in range(len(age_labels)):
                    age_range = age_labels[i]
                    lower_limit, upper_limit = map(int, age_range.split('-'))
                    
                    if lower_limit == 0:
                        age_class_mask = (subset_age_cube.sel(time= '2010-01-01') >= lower_limit) & (subset_age_cube.sel(time= '2010-01-01') < upper_limit+1)
                    else:
                        age_class_mask = (subset_age_cube.sel(time= '2010-01-01') > lower_limit) & (subset_age_cube.sel(time= '2010-01-01') < upper_limit +1)
                        
                    age_class_mask = age_class_mask.where(age_class_mask >0)
                
                    aging_forest = age_class_mask[self.config_file['forest_age_var']].where(aging_forest_class.aging_forest_class==1)
                    diff_aging = diff_age.where(np.isfinite(aging_forest))
                    aging_class_partition = xr.where(diff_aging >= 10, 1, 0).where(np.isfinite(diff_age.age_difference))
                                
                    stand_replaced_forest = age_class_mask[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)
                    diff_replaced = diff_age.where(np.isfinite(stand_replaced_forest))    
                    stand_replaced_class_partition = xr.where(diff_replaced < 10, 1, 0).where(np.isfinite(diff_age.age_difference))
                        
                    if i == len(age_labels) - 1:
                        aging_class_partition = aging_class_partition.expand_dims({"age_class": ['>' + age_range.split('-')[0]]})
                        stand_replaced_class_partition = stand_replaced_class_partition.expand_dims({"age_class": ['>' + age_range.split('-')[0]]})
        
                    else:
                        aging_class_partition = aging_class_partition.expand_dims({"age_class": [age_range]})
                        stand_replaced_class_partition = stand_replaced_class_partition.expand_dims({"age_class": [age_range]})
                        
                    subset_agb_cube = self.agb_cube.sel(IN).sel(members=member_)[['aboveground_biomass']]
                    subset_agb_cube = subset_agb_cube.where(subset_agb_cube>0)
                    diff_agb = subset_agb_cube.sel(time= '2020-01-01') - subset_agb_cube.sel(time= '2010-01-01')
                    
                    stand_replaced_class_partition_member = diff_agb.where(stand_replaced_class_partition.age_difference ==1).rename({'aboveground_biomass': 'stand_replaced'})
                    aging_class_partition_member = diff_agb.where(aging_class_partition.age_difference ==1).rename({'aboveground_biomass': 'gradually_ageing'})
                    out_cube = xr.merge([aging_class_partition_member, stand_replaced_class_partition_member]).expand_dims("members").transpose("members", "age_class", 'latitude', 'longitude')
                      
                    self.agbDiffPartition_cube.CubeWriter(out_cube, n_workers=2)
        
    def BiomassDiffPartitionCubeInit(self) -> None:
        """
        Initialize the biomass difference partition cube.

        Returns
        -------
        None
        """
        
        self.agbDiffPartition_cube.init_variable(self.config_file['cube_variables'])
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync"))
    
    def BiomassDiffPartitionCalc(self,
                     task_id=None) -> None:
        """
        Calculate the fraction of each age class.

        Parameters
        ----------
        task_id : int, optional
            Task ID for parallel processing. Default is None.
        
        Returns
        -------
        None
        """
        lat_chunk_size, lon_chunk_size = self.agbDiffPartition_cube.cube.chunks['latitude'][0], self.agbDiffPartition_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.agbDiffPartition_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.agbDiffPartition_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
            
            lat_values = self.agbDiffPartition_cube.cube.latitude.values[lat_slice]
            lon_values = self.agbDiffPartition_cube.cube.longitude.values[lon_slice]

            # Select the extent based on the slice indices
            selected_extent = {"latitude": slice(lat_values[0], lat_values[-1]), 
                               "longitude": slice(lon_values[0], lon_values[-1])}
            
            # Process the chunk
            self.process_chunk(selected_extent)
        
        else:
           print(f"Task ID {task_id} is out of range. No chunk to process.")
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent) -> None:
        """
        Process a chunk of data.

        Parameters
        ----------
        extent : dict
            Dictionary defining the extent of the chunk to process.
        
        Returns
        -------
        None
        """
        
        self._calc_func(extent).compute()
        
    def ParallelResampling(self, 
                           n_jobs:int=20) -> None:
        """
        Resample the data in parallel.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs to run. Default is 20.
        
        Returns
        -------
        None
        """
        
        member_out = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit a future for each member
            futures = [executor.submit(self.BiomassDiffPartitionResample, member_) 
                       for member_ in np.arange(self.config_file['num_members'])]
            
            # As each future completes, get the result and add it to member_out
            for future in concurrent.futures.as_completed(futures):
                try:
                    member_out.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")

        xr.concat(member_out, dim = 'members').to_zarr(self.config_file['BiomassDiffPartitionResample_cube'] + '_{resolution}deg'.format(resolution = str(self.config_file['target_resolution'])), mode= 'w')
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync"))
        
    def BiomassDiffPartitionResample(self, member_:int=0) -> xr.Dataset:
        """
        Resample biomass difference partition data.

        Parameters
        ----------
        member_ : int, optional
            Member index to process. Default is 0.

        Returns
        -------
        xr.Dataset
            Resampled biomass difference partition data.
        """
                        
        agbDiffPartition_cube = xr.open_zarr(self.config_file['cube_location']).sel(members = member_).drop_vars('members')
        zarr_out_ = []
        
        for var_ in set(agbDiffPartition_cube.variables.keys()) - set(agbDiffPartition_cube.dims):
            
            out = []
            for class_ in agbDiffPartition_cube.age_class.values:
            
                LatChunks = np.array_split(agbDiffPartition_cube.latitude.values, self.config_file['n_chunks'])
                LonChunks = np.array_split(agbDiffPartition_cube.longitude.values, self.config_file['n_chunks'])
                chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
            		        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
            		    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] 
                
                iter_ = 0
                for chunck in chunk_dict:
                    
                    data_chunk = agbDiffPartition_cube[var_].sel(chunck).sel(age_class = class_).transpose('latitude', 'longitude')
                    data_chunk = data_chunk.where(np.isfinite(data_chunk), -9999).astype('float32')  
                    
                    data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                    data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                    data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
                    data_chunk.attrs = {'long_name': 'Biomass difference',
                                        'units': 'Ton /ha',
                                        'valid_max': 300,
                                        'valid_min': -300}
                    data_chunk.attrs["_FillValue"] = -9999  
                    out_dir = '{tmp_folder}/{member}/{var_}/{class_}/'.format(tmp_folder = self.tmp_folder, member= str(member_), var_ = var_, class_ = class_)
                    if not os.path.exists(out_dir):
               		    os.makedirs(out_dir)
                           
                    data_chunk.rio.to_raster(raster_path= out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)), 
                                             driver="COG", BIGTIFF='YES', compress=None,  dtype= 'float32')      
                    
                    gdalwarp_command = [
                                        'gdal_translate',
                                        '-a_nodata', '-9999',
                                        out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)),
                                        out_dir + '{var_}_{iter_}_nodata.tif'.format(var_ = var_, iter_=str(iter_))                
                                    ]
                    subprocess.run(gdalwarp_command, check=True)
                    os.remove(out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)))
                    
                    iter_ += 1
            
                input_files = glob.glob(os.path.join(out_dir, '*_nodata.tif'))
                vrt_filename = out_dir + '/{var_}_{class_}.vrt'.format(var_ = var_, class_= class_)
                    
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
                   out_dir + f'{var_}_{class_}_{self.config_file["target_resolution"]}deg.tif'.format(var_=var_, class_= class_),
                ]
                subprocess.run(gdalwarp_command, check=True)
                
                for file_ in input_files:
                    os.remove(file_)
                    
                da_ =  rio.open_rasterio(out_dir + f'{var_}_{class_}_{self.config_file["target_resolution"]}deg.tif'.format(var_=var_, class_= class_))     
                da_ =  da_.isel(band=0).drop_vars('band').rename({'x': 'longitude', 'y': 'latitude'}).to_dataset(name = var_)
                out.append(da_.assign_coords(age_class= class_))
                
            zarr_out_.append(xr.concat(out, dim = 'age_class').transpose('latitude', 'longitude', 'age_class'))
        
        return xr.merge(zarr_out_).expand_dims({"members": [member_]})
        
        

                
    