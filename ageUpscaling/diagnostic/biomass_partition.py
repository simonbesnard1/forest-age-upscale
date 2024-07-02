#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   biomass_partition.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard.sim@gmail.com
@License :   (C)Copyright 2022-2023, GFZ-Potsdam

This module defines a method class for handling the creation and updating of biomass partition data cubes.

Example usage:
--------------
from biomass_partition import BiomassPartition

# Create a BiomassPartition instance
config_path = 'path/to/config.yml'
study_dir = 'path/to/study_dir'

biomass_partition = BiomassPartition(Config_path=config_path, study_dir=study_dir)
biomass_partition.BiomassPartitionCubeInit()
biomass_partition.BiomassPartitionCalc(task_id=0)
biomass_partition.ParallelBiomassPartitionResampling(n_jobs=20)
biomass_partition.ParallelBiomassTotalResampling(n_jobs=20)
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

class BiomassPartition(ABC):
    """
    BiomassPartition class used for creating and updating biomass partition data cubes.

    Parameters
    ----------
    Config_path : str
        Path to the configuration file.
    study_dir : str, optional
        Path to the study directory. Default is None.
    n_jobs : int, optional
        Number of workers. Default is 1.
    **kwargs : additional keyword arguments
        Additional arguments.
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
        
        sync_file_features = os.path.abspath(f"{study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        
        self.age_cube = xr.open_zarr(self.config_file['ForestAge_cube'], synchronizer=self.sync_feature)
        self.agb_cube = xr.open_zarr(self.config_file['Biomass_cube'], synchronizer=self.sync_feature)
        
        age_class = np.array(self.config_file['age_classes'])
        age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
        age_labels[-1] = '>' + age_labels[-1].split('-')[0]
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        self.config_file['output_writer_params']['dims']['age_class'] = age_labels
        self.config_file['output_writer_params']['dims']['members'] =  self.config_file['num_members']
        
        self.agbPartition_cube = DataCube(cube_config = self.config_file)
        
        self.tmp_folder = os.path.join(self.config_file['tmp_dir'], 'biomassPartition/')
        
    @dask.delayed
    def _calc_func(self, 
                   IN) -> None:
        
        """
        Calculate the fraction of data for each age category based on the age_cube and the age_classes from config_file.

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
                subset_agb_cube = self.agb_cube.sel(IN).sel(members=member_)[['aboveground_biomass']]
                subset_agb_cube = subset_agb_cube.where(subset_agb_cube>0)
                agb_2020 = subset_agb_cube.sel(time= '2020-01-01')
                growth_rate = self.calculate_growth_rate(subset_agb_cube.sel(time= '2020-01-01'), 
                                                         subset_agb_cube.sel(time= '2010-01-01'))
                
                stand_replaced_biomass_member = agb_2020.where(stand_replaced_class.stand_replaced_class ==1).rename({'aboveground_biomass': 'stand_replaced_total'})
                aging_class_biomass_member = agb_2020.where(aging_forest_class.aging_forest_class ==1).rename({'aboveground_biomass': 'gradually_ageing_total'})
                out_cube = xr.merge([stand_replaced_biomass_member, aging_class_biomass_member]).expand_dims("members").transpose("members", 'latitude', 'longitude')
                self.agbPartition_cube.CubeWriter(out_cube, n_workers=2)
            
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
                        
                    stand_replaced_class_partition_member = agb_2020.where(stand_replaced_class_partition.age_difference ==1).rename({'aboveground_biomass': 'stand_replaced'})
                    aging_class_partition_member = agb_2020.where(aging_class_partition.age_difference ==1).rename({'aboveground_biomass': 'gradually_ageing'})
                    stand_replaced_growth_rate_member = growth_rate.where(stand_replaced_class_partition.age_difference ==1).rename({'aboveground_biomass': 'stand_replaced_AGR'})
                    aging_growth_rate_member = growth_rate.where(aging_class_partition.age_difference ==1).rename({'aboveground_biomass': 'gradually_ageing_AGR'})
                    out_cube = xr.merge([aging_class_partition_member, aging_growth_rate_member, stand_replaced_class_partition_member, stand_replaced_growth_rate_member]).expand_dims("members").transpose("members", "age_class", 'latitude', 'longitude')
                      
                    self.agbPartition_cube.CubeWriter(out_cube, n_workers=2)
                
    def BiomassPartitionCubeInit(self) -> None:
        """
        Initialize the biomass partition data cube.
        """
        
        self.agbPartition_cube.init_variable(self.config_file['cube_variables'])
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync"))
    
    def BiomassPartitionCalc(self,
                             task_id=None) -> None:
        """
        Calculate the fraction of each age class.

        Parameters
        ----------
        task_id : int, optional
            Task ID for parallel processing. Default is None.
        """
        lat_chunk_size, lon_chunk_size = self.agbPartition_cube.cube.chunks['latitude'][0], self.agbPartition_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.agbPartition_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.agbPartition_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
            
            lat_values = self.agbPartition_cube.cube.latitude.values[lat_slice]
            lon_values = self.agbPartition_cube.cube.longitude.values[lon_slice]

            # Select the extent based on the slice indices
            selected_extent = {"latitude": slice(lat_values[0], lat_values[-1]), 
                               "longitude": slice(lon_values[0], lon_values[-1])}
            
            # Process the chunk
            self.process_chunk(selected_extent)
        
        else:
           print(f"Task ID {task_id} is out of range. No chunk to process.")

        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync"))
                
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
        
    def ParallelBiomassPartitionResampling(self, 
                                           n_jobs:int=20) -> None:
        """
        Perform parallel biomass partition resampling.

        Parameters
        ----------
        n_jobs : int, optional
            Number of jobs to run in parallel. Default is 20.

        Returns
        -------
        None
        """
        
        member_out = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit a future for each member
            futures = [executor.submit(self.BiomassPartitionResample, member_) 
                       for member_ in np.arange(self.config_file['num_members'])]
            
            # As each future completes, get the result and add it to member_out
            for future in concurrent.futures.as_completed(futures):
                try:
                    member_out.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")

        xr.concat(member_out, dim = 'members').sortby('members').to_zarr(self.config_file['BiomassPartitionResample_cube'] + '_{resolution}deg'.format(resolution = str(self.config_file['resample_resolution'])), mode= 'w')
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync"))
            
        try:
            shutil.rmtree(self.tmp_folder)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
            
    def ParallelBiomassTotalResampling(self, 
                                       n_jobs:int=20) -> None:
        """
        Perform parallel biomass total resampling.

        Parameters
        ----------
        n_jobs : int, optional
            Number of jobs to run in parallel. Default is 20.

        Returns
        -------
        None
        """
        
        member_out = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit a future for each member
            futures = [executor.submit(self.BiomassTotalResample, member_) 
                       for member_ in np.arange(self.config_file['num_members'])]
            
            # As each future completes, get the result and add it to member_out
            for future in concurrent.futures.as_completed(futures):
                try:
                    member_out.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")

        xr.concat(member_out, dim = 'members').sortby('members').to_zarr(self.config_file['BiomassTotalResample_cube'] + '_{resolution}deg'.format(resolution = str(self.config_file['resample_resolution'])), mode= 'w')
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartition_cube_out_sync_{self.task_id}.zarrsync"))
            
        try:
            shutil.rmtree(self.tmp_folder)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
        
    def BiomassPartitionResample(self, member_:int=0) -> xr.Dataset:
        """
        Resample biomass partition data.

        Parameters
        ----------
        member_ : int, optional
            Member index to process. Default is 0.

        Returns
        -------
        xr.Dataset
            Resampled biomass partition data.
        """
                        
        agbPartition_cube = xr.open_zarr(self.config_file['cube_location']).sel(members = member_).drop_vars('members')
        zarr_out_ = []
        
        for var_ in {item for item in set(agbPartition_cube.variables.keys()) - set(agbPartition_cube.dims) if 'total' not in item}:
    
            out = []
            for class_ in agbPartition_cube.age_class.values:
                out_dir = '{tmp_folder}/{member}/{var_}/{class_}/'.format(tmp_folder = self.tmp_folder, member= str(member_), var_ = var_, class_ = class_)
                if not os.path.exists(out_dir):
           		    os.makedirs(out_dir)
                       
                LatChunks = np.array_split(agbPartition_cube.latitude.values, self.config_file['n_chunks'])
                LonChunks = np.array_split(agbPartition_cube.longitude.values, self.config_file['n_chunks'])
                chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
            		        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
            		    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] 
                
                iter_ = 0
                for chunck in chunk_dict:
                    
                    data_chunk = agbPartition_cube[var_].sel(chunck).sel(age_class = class_).transpose('latitude', 'longitude')
                    data_chunk = data_chunk.where(np.isfinite(data_chunk), -9999).astype('float32')  
                    
                    data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                    data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                    data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
                    data_chunk.attrs = {'long_name': 'Biomass difference',
                                        'units': 'Ton /ha',
                                        'valid_max': 300,
                                        'valid_min': -300}
                    data_chunk.attrs["_FillValue"] = -9999  
                    
                           
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
                    '-tr', str(self.config_file['resample_resolution']), str(self.config_file['resample_resolution']),
                    '-t_srs', 'EPSG:4326',
                    '-of', 'Gtiff',
                    '-te', '-180', '-90', '180', '90',
                    '-r', 'average',
                    '-ot', 'Float32',
                    '-co', 'COMPRESS=LZW',
                    '-co', 'BIGTIFF=YES',
                    '-overwrite',
                    f'/{vrt_filename}',
                   out_dir + f'{var_}_{class_}_{self.config_file["resample_resolution"]}deg.tif'.format(var_=var_, class_= class_),
                ]
                subprocess.run(gdalwarp_command, check=True)
                
                for file_ in input_files:
                    os.remove(file_)
                
                da_ =  rio.open_rasterio(out_dir + f'{var_}_{class_}_{self.config_file["resample_resolution"]}deg.tif'.format(var_=var_, class_= class_))     
                da_ =  da_.isel(band=0).drop_vars('band').rename({'x': 'longitude', 'y': 'latitude'}).to_dataset(name = var_)
                out.append(da_.assign_coords(age_class= class_))
                
            zarr_out_.append(xr.concat(out, dim = 'age_class').transpose('latitude', 'longitude', 'age_class'))
    
        return xr.merge(zarr_out_).expand_dims({"members": [member_]})
    
    def BiomassTotalResample(self, member_:int=0) -> xr.Dataset:
        """
        Resample biomass total data.
    
        Parameters
        ----------
        member_ : int, optional
            Member index to process. Default is 0.
    
        Returns
        -------
        xr.Dataset
            Resampled biomass total data.
        """
                        
        agbPartition_cube = xr.open_zarr(self.config_file['cube_location']).sel(members = member_).drop_vars('members')
        zarr_out_ = []
        
        for var_ in {item for item in set(agbPartition_cube.variables.keys()) - set(agbPartition_cube.dims) if 'total' in item}:
    
            LatChunks = np.array_split(agbPartition_cube.latitude.values, self.config_file['n_chunks'])
            LonChunks = np.array_split(agbPartition_cube.longitude.values, self.config_file['n_chunks'])
            chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
        		        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
        		    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] 
            
            iter_ = 0
            for chunck in chunk_dict:
                
                data_chunk = agbPartition_cube[var_].sel(chunck).transpose('latitude', 'longitude')
                data_chunk = data_chunk.where(np.isfinite(data_chunk), -9999).astype('float32')  
                
                data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
                data_chunk.attrs = {'long_name': 'Biomass difference',
                                    'units': 'Ton /ha',
                                    'valid_max': 300,
                                    'valid_min': -300}
                data_chunk.attrs["_FillValue"] = -9999  
                out_dir = '{tmp_folder}/{member}/{var_}/'.format(tmp_folder = self.tmp_folder, member= str(member_), var_ = var_)
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
                '-tr', str(self.config_file['resample_resolution']), str(self.config_file['resample_resolution']),
                '-t_srs', 'EPSG:4326',
                '-of', 'Gtiff',
                '-te', '-180', '-90', '180', '90',
                '-r', 'average',
                '-ot', 'Float32',
                '-co', 'COMPRESS=LZW',
                '-co', 'BIGTIFF=YES',
                '-overwrite',
                f'/{vrt_filename}',
               out_dir + f'{var_}_{self.config_file["resample_resolution"]}deg.tif'.format(var_=var_),
            ]
            subprocess.run(gdalwarp_command, check=True)
            
            for file_ in input_files:
                os.remove(file_)
            
            da_ =  rio.open_rasterio(out_dir + f'{var_}_{self.config_file["resample_resolution"]}deg.tif'.format(var_=var_))     
            da_ =  da_.isel(band=0).drop_vars('band').rename({'x': 'longitude', 'y': 'latitude'}).to_dataset(name = var_)
                
            zarr_out_.append(da_.transpose('latitude', 'longitude'))

        return xr.merge(zarr_out_).expand_dims({"members": [member_]})

    def calculate_growth_rate(self, biomass_2010: xr.DataArray, biomass_2020: xr.DataArray, n_years: int = 10) -> xr.DataArray:
        """
        Calculate the average annual growth rate.

        Parameters
        ----------
        biomass_2010 : xr.DataArray
            Biomass data for the year 2010.
        biomass_2020 : xr.DataArray
            Biomass data for the year 2020.
        n_years : int, optional
            Number of years between the two datasets. Default is 10.

        Returns
        -------
        xr.DataArray
            Average annual growth rate.
        """
        # Calculate the ratio of biomass in 2020 to biomass in 2010
        ratio = biomass_2020 / biomass_2010

        # Calculate the 10th root of the ratio to find the average annual growth rate
        average_annual_growth_rate = ratio ** (1 / n_years) - 1

        return average_annual_growth_rate

        
                
        
                
    