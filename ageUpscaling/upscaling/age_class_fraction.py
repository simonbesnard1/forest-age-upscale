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
import atexit
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

synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

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
        
    def _calc_func(self, 
                   IN) -> None:
        
        subset_age_cube = xr.open_zarr(self.config_file['ForestAge_cube']).sel(IN)
        
        age_class = np.array(self.config_file['age_classes'])
        age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
        
        age_class_fraction = []
        for i in range(len(age_labels)):
            age_range = age_labels[i]
            lower_limit, upper_limit = map(int, age_range.split('-'))
            age_class_mask = (subset_age_cube >= lower_limit) & (subset_age_cube < upper_limit)
            age_class_mask = age_class_mask.where(np.isfinite(subset_age_cube))
            if i == len(age_labels) - 1:
                age_class_mask = age_class_mask.assign_coords(age_class= '>=' + age_range.split('-')[0])
            else:
                age_class_mask = age_class_mask.assign_coords(age_class= age_range)
                
            age_class_fraction.append(age_class_mask)
        age_class_fraction = xr.concat(age_class_fraction, dim = 'age_class')
        
        self.age_class_frac_cube.CubeWriter(age_class_fraction, n_workers=2)
            
    def AgeFractionCalc(self) -> None:
        """Calculate the fraction of each age class.
        
        """
        age_class = np.array(self.config_file['age_classes'])
        age_labels = [f"{age1}-{age2}" for age1, age2 in zip(age_class[:-1], age_class[1:])]
        age_labels[-1] = '>=' + age_labels[-1].split('-')[0]
        self.config_file['output_writer_params']['dims']['latitude'] = xr.open_zarr(self.config_file['ForestAge_cube']).latitude.values
        self.config_file['output_writer_params']['dims']['longitude'] =  xr.open_zarr(self.config_file['ForestAge_cube']).longitude.values
        self.config_file['output_writer_params']['dims']['age_class'] = age_labels
        self.age_class_frac_cube = DataCube(cube_config = self.config_file)
        self.age_class_frac_cube.init_variable(self.config_file['cube_variables'], 
                                               njobs= len(self.config_file['cube_variables'].keys()))
        
        LatChunks = np.array_split(self.config_file['output_writer_params']['dims']['latitude'], self.config_file["num_chunks"])
        LonChunks = np.array_split(self.config_file['output_writer_params']['dims']['longitude'], self.config_file["num_chunks"])
        
        AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                       "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                    for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
        
        if (self.n_jobs > 1):
            
            with dask.config.set({'distributed.worker.threads': self.n_jobs}):

                futures = [self._calc_func(extent) for extent in AllExtents]
                dask.compute(*futures, num_workers=self.n_jobs)
                        
        else:
            for extent in tqdm(AllExtents, desc='Calculating age class fraction'):
                self._calc_func(extent)
                        
        for var_ in self.config_file['cube_variables'].keys():
            out_ = [] 
            
            for class_ in self.age_class_frac_cube.cube.age_class.values:
                 
                data_class = xr.open_zarr(self.config_file['cube_location'])[var_].sel(age_class = class_).transpose('time', 'latitude', 'longitude').astype("int16")         
                data_class.latitude.attrs = {'standard_name': 'latitude', 'units': 'degrees_north', 'crs': 'EPSG:4326'}
                data_class.longitude.attrs = {'standard_name': 'longitude', 'units': 'degrees_east', 'crs': 'EPSG:4326'}
                data_class = data_class.rio.write_crs("epsg:4326", inplace=True)
                data_class.attrs = {'long_name': 'Forest age fraction - {class_}'.format(class_ = class_),
                                    'units': 'adimensional',
                                    'valid_max': 1,
                                    'valid_min': 0}
                LatChunks = np.array_split(data_class.latitude.values, 2)
                LonChunks = np.array_split(data_class.longitude.values, 2)
                chunk_dict = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
            		        "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                              for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
                
                iter_ = 0
                for chunck in chunk_dict:
	
                    data_class.sel(chunck).rio.to_raster(raster_path=self.study_dir + '/age_class_{class_}_{iter_}.tif'.format(class_ =class_, iter_=str(iter_)), driver="COG", BIGTIFF='YES', compress='LZW', dtype="int16")       
                    iter_ += 1
                
                gdalwarp_command = [
                                    'gdalbuildvrt',
                                    self.study_dir + 'age_class_{class_}.vrt'.format(class_=class_),
                                    ] + glob.glob(self.study_dir + '*.tif')
                subprocess.run(gdalwarp_command, check=True)
                
                gdalwarp_command = [
                    'gdalwarp',
                    self.study_dir + 'age_class_{class_}.vrt'.format(class_=class_),
                    self.study_dir + f'age_class_fraction_{self.config_file["target_resolution"]}deg.tif',
                    '-tr', str(self.config_file['target_resolution']), str(self.config_file['target_resolution']),
                    '-t_srs', 'EPSG:4326',
                    '-of', 'Gtiff',
                    '-te', '-180', '-90', '180', '90',
                    '-r', 'average',
                    '-ot', 'Float32',
                    '-srcnodata NaN'
                ]
        
                subprocess.run(gdalwarp_command, check=True)
                tif_files = glob.glob(os.path.join(self.study_dir, 'age_class_{class_}*.tif'.format(class_=class_)))
                for tif_file in tif_files:
                    os.remove(tif_file)
                os.remove(self.study_dir + 'age_class_{class_}.vrt'.format(class_=class_))                    
                da_ =  rio.open_rasterio(self.study_dir + f'age_class_fraction_{self.config_file["target_resolution"]}.tif')     
                    
                da_ =  da_.rename({'x': 'longitude', 'y': 'latitude', 'band': 'time'}).assign_coords(age_class= class_)
                da_['time'] = data_class.time
                out_.append(da_)
                            
            out_ = xr.concat(out_, dim = 'age_class').to_dataset(name = 'forestAge_fraction').transpose('latitude', 'longitude', 'time', 'age_class')
            da_.to_zarr(self.study_dir + 'age_fraction_{var_}_{resolution}'.format(var_ = var_, resolution = str(self.config_file['target_resolution'])), mode= 'w')
                
        tif_files = glob.glob(os.path.join(self.study_dir, '*.tif'))
        for tif_file in tif_files:
            os.remove(tif_file)
        shutil.rmtree(self.config_file['cube_location'])
    
        
        
        

    