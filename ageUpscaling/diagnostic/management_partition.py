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

class ManagementPartition(ABC):
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
        sync_file_features = os.path.abspath(f"{study_dir}/management_features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        self.management_type_cube = xr.open_zarr(self.config_file['ForestManagement_cube'], synchronizer=self.sync_feature).isel(time=0).drop('time')
        self.age_cube = xr.open_zarr(self.config_file['ForestAge_cube'], synchronizer=self.sync_feature)
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/management_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        self.config_file['output_writer_params']['dims']['members'] =  self.config_file['num_members']
        
        self.management_partition_cube = DataCube(cube_config = self.config_file)
        self.tmp_folder = os.path.join(self.config_file['tmp_dir'], 'management_partition/')
        
    @dask.delayed
    def _calc_func(self, 
                   IN) -> None:
        
        """
          Calculate the fraction of data for each age category based on the management_cube and the age_classes from config_file.
        
          Args:
          - IN: Dictionary for subsetting the management_cube.
        
          Returns:
          - age_class_fraction: xarray DataArray with fractions for each age class.
        """
        
       
        for member_ in np.arange(self.config_file['num_members']):
        
            subset_age_cube = self.age_cube.sel(IN).sel(members=member_)[[self.config_file['forest_age_var']]]
            diff_age = subset_age_cube.sel(time= '2020-01-01') - subset_age_cube.sel(time= '2010-01-01')
            diff_age = diff_age.where(diff_age != 0, 10).where(np.isfinite(diff_age))
            
            subset_management_cube = self.management_type_cube.sel(IN)[[self.config_file['management_var']]]
            
            intact_forests = xr.where(subset_management_cube ==11, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'intact_forests'})
            naturally_regenerated = xr.where(subset_management_cube ==20, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'naturally_regenerated'})
            planted_forest = xr.where(subset_management_cube ==31, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'planted_forest'})
            plantation_forest = xr.where(subset_management_cube ==32, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'plantation_forest'})
            oil_palm = xr.where(subset_management_cube ==40, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'oil_palm'})
            agroforestry = xr.where(subset_management_cube ==53, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'agroforestry'})
                
            intact_forests_diff = diff_age.where(intact_forests==1)
            naturally_regenerated_diff = diff_age.where(naturally_regenerated==1)
            planted_forest_diff = diff_age.where(planted_forest==1)
            plantation_forest_diff = diff_age.where(plantation_forest==1)
            oil_palm_diff = diff_age.where(oil_palm==1)
            agroforestry_diff = diff_age.where(agroforestry==1)       
            
            out_cube = xr.merge([intact_forests_diff, naturally_regenerated_diff, planted_forest_diff, plantation_forest_diff, oil_palm_diff, agroforestry_diff]).expand_dims("members").transpose("members",'latitude', 'longitude')     
            self.management_partition_cube.CubeWriter(out_cube, n_workers=6)  

    def ManagementPartitionCubeInit(self):
        
        self.management_partition_cube.init_variable(self.config_file['cube_variables'])
    
    def ManagementPartitionCalc(self,
                     task_id=None) -> None:
        """Calculate the fraction of each age class.
        
        """
        lat_chunk_size, lon_chunk_size = self.management_partition_cube.cube.chunks['latitude'][0], self.management_partition_cube.cube.chunks['longitude'][0]

        # Calculate the number of chunks for each dimension
        num_lat_chunks = np.ceil(len(self.management_partition_cube.cube.latitude) / lat_chunk_size).astype(int)
        num_lon_chunks = np.ceil(len(self.management_partition_cube.cube.longitude) / lon_chunk_size).astype(int)
     
        # Generate all combinations of latitude and longitude chunk indices
        chunk_indices = list(product(range(num_lat_chunks), range(num_lon_chunks)))
     
        if task_id < len(chunk_indices):
            lat_idx, lon_idx = chunk_indices[task_id]
     
            # Calculate slice indices for latitude and longitude
            lat_slice = slice(lat_idx * lat_chunk_size, (lat_idx + 1) * lat_chunk_size)
            lon_slice = slice(lon_idx * lon_chunk_size, (lon_idx + 1) * lon_chunk_size)
            
            lat_values = self.management_partition_cube.cube.latitude.values[lat_slice]
            lon_values = self.management_partition_cube.cube.longitude.values[lon_slice]

            # Select the extent based on the slice indices
            selected_extent = {"latitude": slice(lat_values[0], lat_values[-1]), 
                               "longitude": slice(lon_values[0], lon_values[-1])}
            
            # Process the chunk
            self.process_chunk(selected_extent)
        
        else:
           print(f"Task ID {task_id} is out of range. No chunk to process.")
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/management_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/management_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/management_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/management_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()
        
    def ParallelResampling(self, 
                           n_jobs:int=20):
        
        member_out = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit a future for each member
            futures = [executor.submit(self.ManagementPartitionResample, member_) 
                       for member_ in np.arange(self.config_file['num_members'])]
            
            # As each future completes, get the result and add it to member_out
            for future in concurrent.futures.as_completed(futures):
                try:
                    member_out.append(future.result())
                except Exception as e:
                    print(f"An error occurred: {e}")

        xr.concat(member_out, dim = 'members').to_zarr(self.config_file['ManagementPartitionResample_cube'] + '_{resolution}deg'.format(resolution = str(self.config_file['target_resolution'])), mode= 'w')

    def ManagementPartitionResample(self, member_:int=0) -> None:
        """
            Calculate the age fraction.
            
            This function processes forest age class data using the Global Age Mapping Integration dataset,
            calculating the age fraction distribution changes over time. The results are saved as raster files.
            
            Attributes:
            - age_class_ds: The dataset containing age class information.
            - zarr_out_: An array to store the output data.
            
            The function performs the following operations:
            - Reads age class data.
            - Loops through each variable and age class in the dataset.
            - Transforms and attributes data.
            - Splits the geographical area into chunks.
            - Processes each chunk for each year.
            - Saves processed data as raster files.
            - Merges and converts the output into Zarr format.
        """
                        
        management_partition_cube = xr.open_zarr(self.config_file['cube_location']).sel(members = member_).drop_vars('members')
        zarr_out_ = []
        for var_ in set(management_partition_cube.variables.keys()) - set(management_partition_cube.dims):
                            
            out_dir = '{tmp_folder}/{member}/{var_}/'.format(tmp_folder = self.tmp_folder, member= str(member_), var_ = var_)
            
            if not os.path.exists(out_dir):
       		    os.makedirs(out_dir)            
             
            data_class =management_partition_cube[var_].transpose('latitude', 'longitude')
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
            
        return xr.merge(zarr_out_).expand_dims({"members": [member_]})
        

                
    