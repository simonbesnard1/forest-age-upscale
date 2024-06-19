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

class ManagementType(ABC):
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
        
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/management_cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.management_type_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.management_type_cube.longitude.values
                
        self.management_cube = DataCube(cube_config = self.config_file)
        self.tmp_folder = os.path.join(self.config_file['tmp_dir'], 'managementType/')

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
        
        subset_management_cube = self.management_type_cube.sel(IN)[[self.config_file['management_var']]]
        
        intact_forests = xr.where(subset_management_cube ==11, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'intact_forests'})
        naturally_regenerated = xr.where(subset_management_cube ==20, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'naturally_regenerated'})
        planted_forest = xr.where(subset_management_cube ==31, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'planted_forest'})
        plantation_forest = xr.where(subset_management_cube ==32, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'plantation_forest'})
        oil_palm = xr.where(subset_management_cube ==40, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'oil_palm'})
        agroforestry = xr.where(subset_management_cube ==53, 1, 0).where(np.isfinite(subset_management_cube)).rename({self.config_file['management_var']: 'agroforestry'})
            
        out_cube = xr.merge([intact_forests, naturally_regenerated, planted_forest, plantation_forest, oil_palm, agroforestry])        
        self.management_cube.CubeWriter(out_cube, n_workers=1)  

    def ManagementCubeInit(self):
        
        self.management_cube.init_variable(self.config_file['cube_variables'])
    
    def ManagementCalc(self,
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
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/management_features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/management_features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/management_cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/management_cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()

    def ManagementResample(self) -> None:
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
            
        xr.merge(zarr_out_).to_zarr(self.config_file['ForestManagementResample_cube'] + '_{resolution}deg'.format(resolution = str(self.config_file['resample_resolution'])), mode= 'w')
        
        try:
            shutil.rmtree(self.tmp_folder)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
        