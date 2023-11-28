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
        sync_file_features = os.path.abspath(f"{study_dir}/features_sync_{self.task_id}.zarrsync")        
        if os.path.isdir(sync_file_features):
            shutil.rmtree(sync_file_features)            
        self.sync_feature = zarr.ProcessSynchronizer(sync_file_features)
        self.age_cube = xr.open_zarr(self.config_file['ForestAge_cube'], synchronizer=self.sync_feature)
     
        self.config_file['sync_file_path'] = os.path.abspath(f"{study_dir}/cube_out_sync_{self.task_id}.zarrsync") 
        self.config_file['output_writer_params']['dims']['latitude'] = self.age_cube.latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  self.age_cube.longitude.values
        
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
        
        var_ = 'forest_age_hybrid'
        subset_age_cube = self.age_cube.sel(IN)[[var_]]
     
        diff_age = subset_age_cube.sel(time= '2020-01-01') - subset_age_cube.sel(time= '2010-01-01')
        stand_replaced_age = diff_age.where(diff_age < 0).rename({var_: 'stand_replaced_age'})
        growing_forest_age = diff_age.where(diff_age > 0).rename({var_: 'growing_forest_age'})
        stable_forest_age = diff_age.where(diff_age == 0).rename({var_: 'stable_forest_age'})            
        stand_replaced_class = xr.where(diff_age < 0, 1, 0).where(np.isfinite(diff_age)).rename({var_: 'stand_replaced_class'})
        growing_forest_class = xr.where(diff_age > 0, 1, 0).where(np.isfinite(diff_age)).rename({var_: 'growing_forest_class'})
        stable_forest_class = xr.where(diff_age == 0, 1, 0).where(np.isfinite(diff_age)).rename({var_: 'stable_forest_class'})
        diff_age = diff_age.rename({var_: 'age_difference'})            
        out_cube = xr.merge([diff_age, stand_replaced_age, growing_forest_age, stable_forest_age, 
                             stand_replaced_class, growing_forest_class, stable_forest_class])
        self.age_diff_cube.CubeWriter(out_cube, n_workers=7)
             
    def AgeDiffCubeInit(self):
        
        self.age_diff_cube.init_variable(self.config_file['cube_variables'], 
                                         njobs= len(self.config_file['cube_variables'].keys()))
    
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
                    
        if os.path.exists(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/features_sync_{self.task_id}.zarrsync"))
        
        if os.path.exists(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync")):
            shutil.rmtree(os.path.abspath(f"{self.study_dir}/cube_out_sync_{self.task_id}.zarrsync"))
        
                
    def process_chunk(self, extent):
        
        self._calc_func(extent).compute()
                
    def AgeDiffFractionCalc(self) -> None:
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
                        
        age_diff_ds = xr.open_zarr(self.config_file['cube_location'])
        zarr_out_ = []
        for var_ in self.config_file['cube_variables'].keys():
            
                 
            data_class =age_diff_ds[var_].transpose('latitude', 'longitude')
            data_class.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
            data_class.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
            data_class = data_class.rio.write_crs("epsg:4326", inplace=True)
            data_class.attrs = {'long_name': 'Forest age difference fraction',
                                'units': 'adimensional',
                                'valid_max': 1,
                                'valid_min': 0}
            
            LatChunks = np.array_split(data_class.latitude.values, 2)
            LonChunks = np.array_split(data_class.longitude.values, 2)
            chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
        		           "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                          for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
                
            ds_ = []
            iter_ = 0
            for chunck in chunk_dict:
                
                chunck.update({'time': year_})
                
                if not os.path.exists(self.study_dir + '/age_class_{class_}/'.format(class_ =class_)):
                    os.makedirs(self.study_dir + '/age_class_{class_}/'.format(class_ =class_))
                
                if not os.path.exists(self.study_dir + '/age_class_{class_}/age_class_{class_}_{iter_}.tif'.format(class_ =class_, iter_=str(iter_))):
                    data_chunk = data_class.sel(chunck)
                    #data_chunk = data_chunk.where(data_chunk>=0, -9999)
                    #data_chunk = data_chunk.rio.write_nodata( -9999, encoded=True, inplace=True) 
                    data_chunk.attrs["_FillValue"] = -9999    
                    data_chunk = data_chunk.astype('int16')
                    data_chunk.rio.to_raster(raster_path=self.study_dir + '/age_class_{class_}/age_class_{class_}_{iter_}.tif'.format(class_ =class_, iter_=str(iter_)), 
                                             driver="COG", BIGTIFF='YES', compress=None, dtype="int16")                            
                   
                    gdalwarp_command = [
                                        'gdal_translate',
                                        '-a_nodata', '-9999',
                                        self.study_dir + '/age_class_{class_}/age_class_{class_}_{iter_}.tif'.format(class_ =class_, iter_=str(iter_)),
                                        self.study_dir + '/age_class_{class_}/age_class_{class_}_{iter_}_nodata.tif'.format(class_ =class_, iter_=str(iter_))
                                    
                                    ]
                    subprocess.run(gdalwarp_command, check=True)

                        
                    iter_ += 1
                      
                    input_files = glob.glob(os.path.join(self.study_dir, 'age_class_{class_}/*_nodata.tif'.format(class_=class_)))
                    vrt_filename = self.study_dir + '/age_class_{class_}.vrt'.format(class_=class_)
                    
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
                       self.study_dir + f'/age_class_fraction_{class_}_{self.config_file["target_resolution"]}deg_{year_}.tif'.format(class_=class_, year_= str(year_)),
                    ]
                    subprocess.run(gdalwarp_command, check=True)
                
                    da_ =  rio.open_rasterio(self.study_dir + f'/age_class_fraction_{class_}_{self.config_file["target_resolution"]}deg_{year_}.tif'.format(class_=class_, year_= str(year_)))     
                    da_ =  da_.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'}).to_dataset(name = var_)
                    da_['time'] = [year_]
                    ds_.append(da_)
                    shutil.rmtree(os.path.join(self.study_dir, 'age_class_{class_}'.format(class_=class_)))
                    os.remove(self.study_dir + '/age_class_{class_}.vrt'.format(class_=class_))                    
                    
                out.append(xr.concat(ds_, dim='time').assign_coords(age_class= class_))
                                  
            zarr_out_.append(xr.concat(out, dim = 'age_class').transpose('latitude', 'longitude', 'time', 'age_class'))
            
            tif_files = glob.glob(os.path.join(self.study_dir, 'age_class_*.tif'))
            for tif_file in tif_files:
                  os.remove(tif_file)

        xr.merge(zarr_out_).to_zarr(self.study_dir + '/ForestAge_fraction_{resolution}deg'.format(resolution = str(self.config_file['target_resolution'])), mode= 'w')



    