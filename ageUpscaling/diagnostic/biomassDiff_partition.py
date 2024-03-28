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

import numpy as np
import yaml as yml

import dask

import xarray as xr
import zarr
import rioxarray as rio

from ageUpscaling.core.cube import DataCube

class BiomassDiffPartition(ABC):
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
        
        self.agbDiffPartition_cube = DataCube(cube_config = self.config_file)
        
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
        #mask_qc = self.agb_cube.sel(IN)['quality_check_changes'].sel(time= '2020-01-01')
        
        diff_age = subset_age_cube.sel(time= '2020-01-01') - subset_age_cube.sel(time= '2010-01-01')
        diff_age = diff_age.where(diff_age != 0, 10).where(np.isfinite(diff_age))
        diff_agb = subset_agb_cube.sel(time= '2020-01-01') - subset_agb_cube.sel(time= '2010-01-01')
        #diff_agb = diff_agb.where(mask_qc != 3)
        
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
                    
                stand_replaced_class_partition = diff_agb.where(stand_replaced_class_partition.age_difference ==1).rename({'aboveground_biomass': 'stand_replaced'})
                aging_class_partition = diff_agb.where(aging_class_partition.age_difference ==1).rename({'aboveground_biomass': 'gradually_ageing'})
                out_cube = xr.merge([aging_class_partition, stand_replaced_class_partition]).transpose("age_class", 'latitude', 'longitude').astype('float16')
                self.agbDiffPartition_cube.CubeWriter(out_cube, n_workers=1)
        
    def BiomassDiffPartitionCubeInit(self):
        
        self.agbDiffPartition_cube.init_variable(self.config_file['cube_variables'])
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync"))
    
    def BiomassDiffPartitionCalc(self,
                     task_id=None) -> None:
        """Calculate the fraction of each age class.
        
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
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()
        
    def BiomassDiffPartitionResample(self) -> None:
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
                        
        agbDiffPartition_cube = xr.open_zarr(self.config_file['cube_location'])
        zarr_out_ = []
        
        for var_ in set(agbDiffPartition_cube.variables.keys()) - set(agbDiffPartition_cube.dims):
            
            out = []
            for class_ in agbDiffPartition_cube.age_class.values:
            
                LatChunks = np.array_split(agbDiffPartition_cube.latitude.values, 3)
                LonChunks = np.array_split(agbDiffPartition_cube.longitude.values, 3)
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
                    out_dir = '{study_dir}/tmp/agbDiffPartition/{var_}/{class_}/'.format(study_dir = self.study_dir, var_ = var_, class_ = class_)
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
        
        xr.merge(zarr_out_).to_zarr(self.study_dir + '/BiomassDiffPartition_{resolution}deg'.format(resolution = str(self.config_file['target_resolution'])), mode= 'w')
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/agbPartitionDiff_cube_out_sync_{self.task_id}.zarrsync"))
        
        try:
            var_path = os.path.join(self.study_dir, 'tmp/agbDiffPartition')
            shutil.rmtree(var_path)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
    

                
    