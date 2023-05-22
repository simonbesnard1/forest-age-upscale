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

import numpy as np
import yaml as yml
import pickle

import dask
import xarray as xr
import zarr
import dask.array as da
from shapely.geometry import Polygon

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

from sklearn.model_selection import train_test_split
import xgboost as xgb

from ageUpscaling.core.cube import DataCube
from ageUpscaling.transformers.spatial import interpolate_worlClim
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.methods.xgboost import XGBoost
from ageUpscaling.methods.RandomForest import RandomForest
from ageUpscaling.methods.autoML import TPOT
from ageUpscaling.methods.feature_selection import FeatureSelection

synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class UpscaleAge(ABC):
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
                 DataConfig_path: str,
                 cube_config_path: str,            
                 base_dir: str,
                 algorithm: str = 'MLP',
                 exp_name: str = None,
                 study_dir: str = None,
                 n_jobs_training: int = 1,
                 n_jobs_upscaling: int = 1,
                 **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
        
        self.algorithm = algorithm
        self.base_dir = base_dir
        self.exp_name = exp_name
        
        if study_dir is None:
            study_dir = self.version_dir(self.base_dir, self.exp_name, self.algorithm)
            os.makedirs(study_dir, exist_ok=False)
        else:
            if not os.path.exists(study_dir):
                raise ValueError(f'restore path does not exist:\n{study_dir}')

        self.study_dir = study_dir
        self.n_jobs_training = n_jobs_training
        self.n_jobs_upscaling = n_jobs_upscaling        
        self.valid_fraction= self.DataConfig["valid_fraction"]
        self.feature_selection= self.DataConfig["feature_selection"]
        self.feature_selection_method= self.DataConfig["feature_selection_method"]      
        self.cube_config['cube_location'] = os.path.join(self.study_dir, self.cube_config['cube_name'])
        self.xval_index_path = self.DataConfig["xval_index_path"]
    
    def version_dir(self, 
                    base_dir: str,
                    exp_name:str,
                    algorithm: str) -> str:
        """Creates a new version of a directory by appending the version number to the end of the directory name.
    
        If the directory already exists, it will be renamed to include the version number before the new directory is created.
        
        Parameters
        ----------
        base_dir : str
            The base directory where the new version of the study directory will be created.
        algorithm : str
            The name of the study directory.
            
        Returns
        -------
        str
            The full path to the new version of the study directory.
        """
        
        return self.increment_dir_version(base_dir,exp_name, algorithm)
    
    @staticmethod
    def increment_dir_version(base_dir: str,
                              exp_name:str,
                              algorithm:str) -> str:
        """Increments the version of a directory by appending the next available version number to the end of the directory name.
        
        Parameters
        ----------
        base_dir : str
            The base directory for the study.
        algorithm : str
            The name of the study.
        
        Returns
        -------
        str
            The name of the new directory with the incremented version number.
        """
        if not os.path.isdir(os.path.join(base_dir, exp_name, algorithm)):
            os.makedirs(os.path.join(base_dir, exp_name, algorithm))
        
        dir_list = [d for d in os.listdir(os.path.join(base_dir, exp_name, algorithm)) if d.startswith("version")]
        
        dir_list.sort()
        
        if len(dir_list) == 0:
            version = "1.0"
        else:
            last_dir = dir_list[-1]
            
            _, version = last_dir.split("-")
            
            major, minor = version.split(".")
            major = int(major)
            minor = int(minor)
            minor += 1
            if minor >= 10:
                major += 1
                minor = 0
            version = f"{major}.{minor}"
        
        return f"{base_dir}/{exp_name}/{algorithm}/version-{version}"
    
    @dask.delayed
    def _predict_func(self, 
                      IN) -> None:
        
        lat_start, lat_stop = IN['latitude'].start, IN['latitude'].stop
        lon_start, lon_stop = IN['longitude'].start, IN['longitude'].stop
        buffer_IN = Polygon([(lon_start, lat_start), (lon_start, lat_stop),(lon_stop, lat_stop), (lon_stop, lat_start)]).buffer(0.01)
        buffer_IN = {'latitude': slice(buffer_IN.bounds[3], buffer_IN.bounds[1], None),
                     'longitude': slice(buffer_IN.bounds[0], buffer_IN.bounds[2], None)}
        
        var_selected           = self.best_models['Classifier']['selected_features'] + self.best_models['Regressor']['selected_features']
        subset_agb_cube        = xr.open_zarr(self.DataConfig['agb_cube'], synchronizer=synchronizer).sel(buffer_IN).astype('float16')
        if not self.cube_config["high_res_pred"]:
            subset_agb_cube    = subset_agb_cube.rename({'agb_001deg_cc_min_{tree_cover}'.format(tree_cover = self.tree_cover) : 'agb'})
       
        subset_agb_cube        = subset_agb_cube.agb.where(subset_agb_cube.agb >0).to_dataset()
        
        if not np.isnan(subset_agb_cube.to_array().values).all():
            
            subset_clim_cube       = xr.open_zarr(self.DataConfig['clim_cube'], synchronizer=synchronizer).sel(buffer_IN)[[x for x in var_selected if "WorlClim" in x]].astype('float16')
           
            if self.cube_config["high_res_pred"]:    
                subset_clim_cube =  interpolate_worlClim(source_ds = subset_clim_cube, target_ds = subset_agb_cube)
            
            subset_clim_cube = subset_clim_cube.expand_dims({'time': subset_agb_cube.time.values}, axis=list(subset_agb_cube.dims).index('time'))
            subset_cube      = xr.merge([subset_agb_cube.sel(IN), subset_clim_cube.sel(IN)])
            
            X_upscale_class = []
            for var_name in self.best_models['Classifier']['selected_features']:
                if self.algorithm == "MLP":
                    X_upscale_class.append(self.norm(subset_cube[var_name], self.best_models['Classifier']['norm_stats'][var_name]))
                else:
                    X_upscale_class.append(subset_cube[var_name])
                
            X_upscale_reg = []
            for var_name in self.best_models['Regressor']['selected_features']:
                if self.algorithm == "MLP":
                    X_upscale_reg.append(self.norm(subset_cube[var_name], self.best_models['Regressor']['norm_stats'][var_name]))
                else:
                    X_upscale_reg.append(subset_cube[var_name])
            
            X_upscale_reg_flattened = []
    
            for arr in X_upscale_reg:
                data = arr.data.flatten()
                X_upscale_reg_flattened.append(data)
                
            X_upscale_class_flattened = []
    
            for arr in X_upscale_class:
                data = arr.data.flatten()
                X_upscale_class_flattened.append(data)
        
            X_upscale_reg_flattened = da.array(X_upscale_reg_flattened).transpose().compute()
            X_upscale_class_flattened = da.array(X_upscale_class_flattened).transpose().compute()
            
            RF_pred_class = np.zeros(X_upscale_reg_flattened.shape[0]) * np.nan
            RF_pred_reg = np.zeros(X_upscale_reg_flattened.shape[0]) * np.nan
            
            mask = (np.all(np.isfinite(X_upscale_reg_flattened), axis=1)) & (np.all(np.isfinite(X_upscale_class_flattened), axis=1))
            
            if (X_upscale_class_flattened[mask].shape[0]>0):
                
                if self.algorithm == "XGBoost":
                    dpred =  xgb.DMatrix(X_upscale_class_flattened[mask])
                    pred_class = np.rint(self.best_models["Classifier"]['best_model'].predict(dpred))
                
                else:
                    pred_class = self.best_models["Classifier"]['best_model'].predict(X_upscale_class_flattened[mask])
                    
                RF_pred_class[mask] = pred_class
                
                if self.algorithm == "XGBoost":
                    dpred =  xgb.DMatrix(X_upscale_reg_flattened[mask])
                    pred_reg= self.best_models["Regressor"]['best_model'].predict(dpred)
                
                else:
                    # pred_reg= self.denorm_target(self.best_models["Regressor"]['best_model'].predict(X_upscale_reg_flattened[mask]), 
                    #                              self.best_models["Regressor"]['norm_stats']['age'])
                    pred_reg= self.best_models["Regressor"]['best_model'].predict(X_upscale_reg_flattened[mask])
                
                pred_reg[pred_reg>=self.DataConfig['max_forest_age'][0]] = self.DataConfig['max_forest_age'][0] -1
                pred_reg[pred_reg<1] = 1
                RF_pred_reg[mask] = pred_reg
                RF_pred_reg[RF_pred_class==1] = self.DataConfig['max_forest_age'][0]            
                out_reg = RF_pred_reg.reshape(len(subset_cube.latitude), len(subset_cube.longitude), len(subset_cube.time), 1)
                output_reg_xr = xr.DataArray(out_reg, 
                                              coords={"latitude": subset_cube.latitude, 
                                                      "longitude": subset_cube.longitude,
                                                      "time": subset_cube.time,                                                          
                                                      'members': [self.member]}, 
                                              dims=["latitude", "longitude", "time", "members"]).to_dataset(name="forest_age_TC{tree_cover}".format(tree_cover= self.tree_cover))
                        
                self.pred_cube.update_cube(output_reg_xr, initialize=False)
        
    def model_tuning(self,
                     run_: int=1,
                     task_:str='Regressor',
                     feature_selection:bool=True,
                     feature_selection_method:str = 'recursive',                     
                     train_subset:dict ={},
                     valid_subset:dict ={}) -> None:
        """Perform model tuning using cross-validation.

        Parameters
        ----------
        run_ : int, optional
            Number of model runs. Default is 1.
        method : str, optional
            The type of model to use for training. Default is 'MLPRegressor'.
        
        Returns
        -------
        None
            The function does not return any values, but it updates the `self.best_model` attribute
            with the best model found during the tuning process.
        """
        if feature_selection:
            self.DataConfig['features_selected'] = FeatureSelection(method=task_, 
                                                                   feature_selection_method = feature_selection_method, 
                                                                   features = self.DataConfig['features'],
                                                                   data = xr.open_dataset(self.DataConfig['training_dataset'])).get_features()
                
        else: 
            self.DataConfig['features_selected'] = self.DataConfig['features'].copy()
            
        if self.algorithm == "MLP":
            
            ml_method = MLPmethod(tune_dir=os.path.join(self.study_dir, "tune"), 
                                   DataConfig= self.DataConfig,
                                   method=self.algorithm + task_)
        elif self.algorithm == "XGBoost":
            ml_method = XGBoost(tune_dir=os.path.join(self.study_dir, "tune"), 
                                DataConfig= self.DataConfig,
                                method=self.algorithm + task_)
        elif self.algorithm == "RandomForest":
            ml_method = RandomForest(tune_dir=os.path.join(self.study_dir, "tune"), 
                                     DataConfig= self.DataConfig,
                                     method=self.algorithm + task_)
        elif self.algorithm == "TPOT":
            ml_method = TPOT(tune_dir=os.path.join(self.study_dir, "tune"), 
                             DataConfig= self.DataConfig,
                             method=self.algorithm + task_)
        
        ml_method.train(train_subset=train_subset,
                        valid_subset=valid_subset,
                        n_jobs = self.n_jobs_training)
        
        if not os.path.exists(self.study_dir + '/save_model/'):
             os.makedirs(self.study_dir + '/save_model/')
             
        with open(self.study_dir + "/save_model/best_{method}_run{id_}.pickle".format(method = task_, id_ = run_), "wb") as fout:
            pickle.dump(ml_method.best_model, fout)
         
        return {'best_model': ml_method.best_model, 
                'selected_features': self.DataConfig['features_selected'], 
                'norm_stats' : ml_method.mldata.norm_stats}
    
    def ForwardRun(self) -> None:
        """Perform forward run of the model, which consists of generating high resolution maps of age using the trained model.

        Parameters
        ----------
        tree_cover_tresholds : dict[str, Any], optional
            Dictionary of tree cover tresholds to use for the forward run, default is {'000', '005', '010', '015', '020', '030'}
        nLatChunks : int, optional
            Number of chunks to use in the latitude dimension, default is 50
        nLonChunks : int, optional
            Number of chunks to use in the longitude dimension, default is 50
        high_res_pred : bool, optional
            Boolean indicating whether to perform high resolution prediction, default is False
        """
        
        self.pred_cube = DataCube(cube_config = self.cube_config)
        self.pred_cube.init_variable(self.cube_config['cube_variables'], 
                                      njobs= len(self.cube_config['cube_variables'].keys()))
        
        cluster_ = np.load(self.xval_index_path)        
        train_subset, valid_subset = train_test_split(cluster_, test_size=self.DataConfig['valid_fraction'], shuffle=True)
        
        for run_ in tqdm(np.arange(self.cube_config['output_writer_params']['dims']['members']), desc='Forward run model members'):
            
            self.member = run_
            self.best_models = {}
            for task_ in ["Regressor", "Classifier"]:
                model_tuned      = self.model_tuning(run_ = run_, 
                                                      task_ = task_,
                                                      feature_selection= self.DataConfig['feature_selection'],
                                                      feature_selection_method = self.DataConfig['feature_selection_method'],     
                                                      train_subset=train_subset, 
                                                      valid_subset=valid_subset)
                self.best_models[task_] = model_tuned      
            
            for tree_cover in self.cube_config["tree_cover_tresholds"]:
                
                if (self.cube_config["high_res_pred"] and tree_cover != '000'):
                    raise ValueError(f'tree cover threshold of {tree_cover} is not supported for the high-resolution cubes -  Thereshold has to be 000')
                
                self.tree_cover = tree_cover
                LatChunks = np.array_split(self.pred_cube.cube.latitude.values, self.cube_config["num_chunks"])
                LonChunks = np.array_split(self.pred_cube.cube.longitude.values, self.cube_config["num_chunks"])
                
                AllExtents = [{"latitude":slice(LatChunks[lat][0], LatChunks[lat][-1]),
                                "longitude":slice(LonChunks[lon][0], LonChunks[lon][-1])} 
                            for lat, lon in product(range(len(LatChunks)), range(len(LonChunks)))]
            
                if (self.n_jobs_upscaling > 1):
                    
                    cluster = SLURMCluster(queue= str(os.environ.get('SLURM_JOB_PARTITION')),
                                            account= str(os.environ.get('SLURM_JOB_USER')),
                                            cores= int(os.environ.get('SLURM_CPUS_PER_TASK')),
                                            memory= str(os.environ.get('SLURM_MEM_PER_NODE')),
                                            job_extra_directives=['--nodes={n_nodes}'.format(n_nodes = str(os.environ.get('SLURM_JOB_NUM_NODES'))), 
                                                                  '--ntasks-per-node={ntasks_}'.format(ntasks_= str(int(os.environ.get('SLURM_TASKS_PER_NODE').split('(x')[0])))])
                    
                    #cluster.scale(jobs=10) 
                    client = Client(cluster)
                    futures = client.map(self._predict_func, AllExtents)
                    _ = client.gather(futures)
            
                    # with dask.config.set({'distributed.worker.memory.target': 50*1024*1024*1024, 
                    #                       'distributed.worker.threads': 2}):

                    #     futures = [self._predict_func(i) for i in AllExtents]
                    #     dask.compute(*futures, num_workers=self.n_jobs_upscaling)    
                    cluster.close()
                    client.close()
                else:
                    for extent in AllExtents:
                        self._predict_func(extent).compute()
            
            shutil.rmtree(os.path.join(self.study_dir, "tune"))
                            
    def norm(self, 
             x: np.array,
             norm_stats:dict) -> np.array:
        """Normalize an array of values using the given normalization statistics.

        Parameters
        ----------
        x : np.array
            The array of values to normalize.
        norm_stats : dict
            A dictionary containing the normalization statistics, with keys 'mean' and 'std'.
    
        Returns
        -------
        np.array
            The normalized array of values.
        """
        
        return (x - norm_stats['mean']) / norm_stats['std'] 
    
    def denorm_target(self, 
                      x: np.array,
                      norm_stats:dict) -> np.array:
        """De-normalize an array of values using the given normalization statistics.

        Parameters
        ----------
        x : np.array
            The array of values to de-normalize.
        norm_stats : dict
            A dictionary containing the normalization statistics, with keys 'mean' and 'std'.
    
        Returns
        -------
        np.array
            The de-normalized array of values.
        """
        
        return x * norm_stats['std'] + norm_stats['mean']
    
