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

class AgeFraction(ABC):
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
        sync_file_features = os.path.abspath(f"{study_dir}/ageClassFrac_features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        self.age_cube = xr.open_zarr(self.config_file['ForestAge_cube'], synchronizer=self.sync_feature)
     
        age_class = np.array(self.config_file['age_classes'])
        age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
        age_labels[-1] = '>' + age_labels[-1].split('-')[0]
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/ageClassFrac_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        self.config_file['output_writer_params']['dims']['age_class'] = age_labels
        self.config_file['output_writer_params']['dims']['members'] =  self.config_file['num_members']
        
        self.age_class_frac_cube = DataCube(cube_config = self.config_file)
        self.tmp_folder = os.path.join(self.config_file['tmp_dir'], 'ageClassFraction/')
        
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
       
        for member_ in np.arange(self.config_file['num_members']):
        
            subset_age_cube = self.age_cube.sel(IN).sel(members=member_)[[self.config_file['forest_age_var']]]
    
            age_class = np.array(self.config_file['age_classes'])
            age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
            
            for i in range(len(age_labels)):
                age_range = age_labels[i]
                lower_limit, upper_limit = map(int, age_range.split('-'))
                if lower_limit == 0:
                    age_class_mask = (subset_age_cube >= lower_limit) & (subset_age_cube < upper_limit+1)
                else:
                    age_class_mask = (subset_age_cube > lower_limit) & (subset_age_cube < upper_limit +1)
                    
                age_class_mask = age_class_mask.where(np.isfinite(subset_age_cube))
                age_class_mask = age_class_mask.where(np.isfinite(age_class_mask), -9999)        
                
                if i == len(age_labels) - 1:
                    age_class_mask = age_class_mask.expand_dims({"members": [member_], "age_class": ['>' + age_range.split('-')[0]]}).transpose('members', "age_class", 'latitude', 'longitude', 'time')

                else:
                    age_class_mask = age_class_mask.expand_dims({"members": [member_], "age_class": [age_range]}).transpose('members', "age_class", 'latitude', 'longitude', 'time')

                self.age_class_frac_cube.CubeWriter(age_class_mask, n_workers=2)
             
    def AgeClassCubeInit(self):
        
        self.age_class_frac_cube.init_variable(self.config_file['cube_variables'])
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/ageClassFrac_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/ageClassFrac_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/ageClassFrac_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/ageClassFrac_cube_out_sync_{self.task_id}.zarrsync"))
     
    def AgeClassCalc(self,
                     task_id=None) -> None:
        """Calculate the fraction of each age class.
        
        """
        
        lat_chunk_size, lon_chunk_size = self.age_class_frac_cube.cube.chunks['latitude'][0], self.age_class_frac_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.age_class_frac_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.age_class_frac_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
            
            lat_values = self.age_class_frac_cube.cube.latitude.values[lat_slice]
            lon_values = self.age_class_frac_cube.cube.longitude.values[lon_slice]

            # Select the extent based on the slice indices
            selected_extent = {"latitude": slice(lat_values[0], lat_values[-1]), 
                               "longitude": slice(lon_values[0], lon_values[-1])}
            
            # Process the chunk
            self.process_chunk(selected_extent)
        
        else:
           print(f"Task ID {task_id} is out of range. No chunk to process.")
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/ageClassFrac_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/ageClassFrac_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/ageClassFrac_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/ageClassFrac_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()
        
    def ParallelResampling(self, n_jobs: int = 20):
        member_out = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit a future for each member
            futures = [executor.submit(self.AgeFractionCalc, member_)
                       for member_ in np.arange(self.config_file['num_members'])]
    
            for future in concurrent.futures.as_completed(futures):
                try:
                    member_out.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")
    
        
        combined_data = self.merge_datasets_by_resolution(member_out)
        for resolution_ in self.config_file['target_resolution']:
            
            combined_data[str(resolution_)].to_zarr(self.config_file['AgeClassResample_cube'] + f'_{resolution_}deg', mode='w')

        # Clean up sync files
        for sync_file in [f"ageClassFrac_features_sync_{self.task_id}.zarrsync",
                          f"ageClassFrac_cube_out_sync_{self.task_id}.zarrsync"]:
            sync_path = os.path.abspath(f"{self.study_dir}/{sync_file}")
            if os.path.exists(sync_path):
                shutil.rmtree(sync_path)
        
        try:
            shutil.rmtree(self.tmp_folder)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")

    def AgeFractionCalc(self, member_: int = 0) -> dict:
        """
        Calculate the age fraction.
        
        This function processes forest age class data using the Global Age Mapping Integration dataset,
        calculating the age fraction distribution changes over time. The results are saved as raster files.
        
        Attributes:
        - age_class_ds: The dataset containing age class information.
        - zarr_out_: An array to store the output data.
        """
        age_class_ds = xr.open_zarr(self.config_file['cube_location']).sel(members=member_).drop_vars('members')
        zarr_out_ = []
    
        for var_ in self.config_file['cube_variables'].keys():
            out = {}
            for class_ in age_class_ds.age_class.values:
                data_class = age_class_ds[var_].sel(age_class=class_).transpose('time', 'latitude', 'longitude')
                data_class.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                data_class.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                data_class = data_class.rio.write_crs("epsg:4326", inplace=True)
                data_class.attrs = {'long_name': f'Forest age fraction - {class_}',
                                    'units': 'adimensional',
                                    'valid_max': 1,
                                    'valid_min': 0}
    
                LatChunks = np.array_split(data_class.latitude.values, self.config_file['n_chunks'])
                LonChunks = np.array_split(data_class.longitude.values, self.config_file['n_chunks'])
                chunk_dict = [{"latitude": slice(LatChunks[lat][0], LatChunks[lat][-1]),
                               "longitude": slice(LonChunks[lon][0], LonChunks[lon][-1])}
                              for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
    
                out_dir = f'{self.tmp_folder}/{member_}/{var_}/{class_}/'
                os.makedirs(out_dir, exist_ok=True)
    
                ds_ = []
                for year_ in data_class.time.values:
                    iter_ = 0
                    for chunck in chunk_dict:
                        chunck.update({'time': year_})
                        data_chunk = data_class.sel(chunck)
                        data_chunk = data_chunk.where(np.isfinite(data_chunk), -9999).astype('int16')
    
                        data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                        data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                        data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
                        data_chunk.attrs["_FillValue"] = -9999
                        data_chunk.rio.to_raster(raster_path=out_dir + f'{var_}_{iter_}.tif',
                                                  driver="COG", BIGTIFF='YES', compress=None, dtype="int16")
    
                        gdalwarp_command = [
                            'gdal_translate',
                            '-a_nodata', '-9999',
                            out_dir + f'{var_}_{iter_}.tif',
                            out_dir + f'{var_}_{iter_}_nodata.tif'
                        ]
                        subprocess.run(gdalwarp_command, check=True)
                        os.remove(out_dir + f'{var_}_{iter_}.tif')
    
                        iter_ += 1
    
                    input_files = glob.glob(os.path.join(out_dir, '*_nodata.tif'))
                    vrt_filename = out_dir + f'/{var_}_{class_}.vrt'
    
                    gdalbuildvrt_command = [
                        'gdalbuildvrt',
                        vrt_filename
                    ] + input_files
    
                    subprocess.run(gdalbuildvrt_command, check=True)
    
                    for resolution_ in self.config_file['target_resolution']:
                        gdalwarp_command = [
                            'gdalwarp',
                            '-srcnodata', '-9999',
                            '-dstnodata', '-9999',
                            '-tr', str(resolution_), str(resolution_),
                            '-t_srs', 'EPSG:4326',
                            '-of', 'Gtiff',
                            '-te', '-180', '-90', '180', '90',
                            '-r', 'average',
                            '-ot', 'Float32',
                            '-co', 'COMPRESS=LZW',
                            '-co', 'BIGTIFF=YES',
                            '-overwrite',
                            vrt_filename,
                            out_dir + f'{var_}_{class_}_{resolution_}deg_{year_}.tif'
                        ]
                        subprocess.run(gdalwarp_command, check=True)
    
                        da_ = rio.open_rasterio(out_dir + f'{var_}_{class_}_{resolution_}deg_{year_}.tif')
                        da_ = da_.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'}).to_dataset(name=var_)
                        da_['time'] = [year_]
                        ds_.append({str(resolution_): {year_: da_}})

                    for file_ in input_files:
                        os.remove(file_)

                combined_datasets = {}
                for dataset in ds_:
                    for resolution, data_dict in dataset.items():
                        if resolution not in combined_datasets:
                            combined_datasets[resolution] = []
    
                        for date, data in data_dict.items():
                            combined_datasets[resolution].append(data)
    
                for resolution, data_list in combined_datasets.items():
                    combined_data = xr.concat(data_list, dim='time').assign_coords(age_class=class_)
                    if resolution not in out:
                        out[resolution] = {}
                    out[resolution][class_] = combined_data
                    
    
        for resolution_ in self.config_file['target_resolution']:
            datasets = []

            for age_class, dataset in out[str(resolution_)].items():
                datasets.append(dataset)
            
            zarr_out_.append({str(resolution_): xr.concat(datasets, dim='age_class').transpose('age_class', 'latitude', 'longitude', 'time').expand_dims({"members": [member_]})})
    
        return zarr_out_
    
    def merge_datasets_by_resolution(self, datasets_by_resolution):
        merged_datasets = {}
        
        for group in datasets_by_resolution:
            for res_dict in group:
                for resolution, dataset in res_dict.items():
                    if resolution not in merged_datasets:
                        merged_datasets[resolution] = []
                    merged_datasets[resolution].append(dataset)
        
        for resolution, dataset_list in merged_datasets.items():
            merged_datasets[resolution] = xr.concat(dataset_list, dim='members')
        
        return merged_datasets
        