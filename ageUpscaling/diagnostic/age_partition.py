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

import dask

import xarray as xr
import zarr
import rioxarray as rio

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
        
        age_class = np.array(self.config_file['age_classes'])
        age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
        age_labels[-1] = '>' + age_labels[-1].split('-')[0]
             
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/ageDiff_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        self.config_file['output_writer_params']['dims']['age_class'] = age_labels
                
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
        out_cube = xr.merge([diff_age, stand_replaced_age, aging_forest_age, stand_replaced_class, aging_forest_class])        
        self.age_diff_cube.CubeWriter(out_cube, n_workers=1)  

        age_class = np.array(self.config_file['age_classes'])
        age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]

        for i in range(len(age_labels)):
            age_range = age_labels[i]
            lower_limit, upper_limit = map(int, age_range.split('-'))
            
            if lower_limit == 0:
                age_class_mask = (subset_age_cube.sel(time= '2010-01-01') >= lower_limit) & (subset_age_cube.sel(time= '2010-01-01') < upper_limit+1)
            else:
                age_class_mask = (subset_age_cube.sel(time= '2010-01-01') > lower_limit) & (subset_age_cube.sel(time= '2010-01-01') < upper_limit +1)
                
            age_class_mask = age_class_mask.where(np.isfinite(subset_age_cube.sel(time= '2010-01-01')))
            
            aging_forest = age_class_mask[self.config_file['forest_age_var']].where(aging_forest_class.aging_forest_class==1)
            diff_aging = diff_age.where(np.isfinite(aging_forest))
            aging_class_partition = xr.where(diff_aging >= 0, 1, 0).where(np.isfinite(diff_age.age_difference))
                        
            stand_replaced_forest = age_class_mask[self.config_file['forest_age_var']].where(stand_replaced_class.stand_replaced_class==1)
            diff_replaced = diff_age.where(np.isfinite(stand_replaced_forest))    
            if lower_limit == 0:
                stand_replaced_class_partition = xr.where(diff_replaced < 10, 1, 0).where(np.isfinite(diff_age.age_difference))
            else:
                stand_replaced_class_partition = xr.where(diff_replaced < 0, 1, 0).where(np.isfinite(diff_age.age_difference))
                
            if i == len(age_labels) - 1:
                aging_class_partition = aging_class_partition.expand_dims({"age_class": ['>' + age_range.split('-')[0]]}).transpose("age_class", 'latitude', 'longitude')
                stand_replaced_class_partition = stand_replaced_class_partition.expand_dims({"age_class": ['>' + age_range.split('-')[0]]}).transpose("age_class", 'latitude', 'longitude')

            else:
                aging_class_partition = aging_class_partition.expand_dims({"age_class": [age_range]}).transpose("age_class", 'latitude', 'longitude')
                stand_replaced_class_partition = stand_replaced_class_partition.expand_dims({"age_class": [age_range]}).transpose("age_class", 'latitude', 'longitude')
                
            stand_replaced_class_partition = stand_replaced_class_partition.rename({'age_difference': 'stand_replaced_class_partition'})
            aging_class_partition = aging_class_partition.rename({'age_difference': 'aging_class_partition'})
            out_cube = xr.merge([aging_class_partition, stand_replaced_class_partition]).drop('time')    
          
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
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/ageDiff_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/ageDiff_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/ageDiff_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/ageDiff_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()

    def AgeDiffResample(self) -> None:
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
                        
        age_diff_cube = xr.open_zarr(self.config_file['cube_location'])
        zarr_out_ = []
        
        for var_ in set(age_diff_cube.variables.keys()) - set(age_diff_cube.dims):
            
            LatChunks = np.array_split(age_diff_cube.latitude.values, 3)
            LonChunks = np.array_split(age_diff_cube.longitude.values, 3)
            chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
        		        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
        		    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))] 
            
            iter_ = 0
            for chunck in chunk_dict:
                
                data_chunk = age_diff_cube[var_].sel(chunck).transpose('latitude', 'longitude')
                data_chunk = data_chunk.where(np.isfinite(data_chunk), -9999).astype("int16")     
                
                data_chunk.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                data_chunk.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                data_chunk = data_chunk.rio.write_crs("epsg:4326", inplace=True)
                data_chunk.attrs = {'long_name': 'Age difference',
                                    'units': 'adimensional',
                                    'valid_max': 1,
                                    'valid_min': 0}
                data_chunk.attrs["_FillValue"] = -9999  
                out_dir = '{study_dir}/tmp/{var_}/'.format(study_dir = self.study_dir, var_ = var_)
                if not os.path.exists(out_dir):
           		    os.makedirs(out_dir)
                       
                data_chunk.rio.to_raster(raster_path= out_dir + '{var_}_{iter_}.tif'.format(var_ = var_, iter_=str(iter_)), 
                                         driver="COG", BIGTIFF='YES', compress=None, dtype="int16")      
                
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
               self.study_dir + f'/{var_}_{self.config_file["target_resolution"]}deg.tif'.format(var_=var_),
            ]
            subprocess.run(gdalwarp_command, check=True)
            
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
                
            da_ =  rio.open_rasterio(self.study_dir + f'/{var_}_{self.config_file["target_resolution"]}deg.tif'.format(var_=var_))     
            da_ =  da_.rename({'x': 'longitude', 'y': 'latitude'}).to_dataset(name = var_)
                
            zarr_out_.append(da_)
        
        xr.merge(zarr_out_).to_zarr(self.study_dir + '/AgeDiff_{resolution}deg'.format(resolution = str(self.config_file['target_resolution'])), mode= 'w')
        
        tif_files = glob.glob(os.path.join(self.study_dir, '*.tif'))
        for tif_file in tif_files:
              os.remove(tif_file)

                
    