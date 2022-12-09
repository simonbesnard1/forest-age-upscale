#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:13:19 2019

#gdalwarp -ts 2160 4320 -dstnodata -9999 -r average -of netCDF /minerva/BGI/scratch/sbesnard/age_upscale/age_product/age_MPI_product_TC020.nc /minerva/BGI/scratch/sbesnard/age_upscale/age_product/age_MPI_product_agg_TC020.nc

@author: sbesnard
"""
import numpy as np
import xarray as xr
import os
import yaml as yml
import abc as ABC
import tqdm
from sklearn.model_selection import train_test_split
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.core.cube import DataCube
import shutil
import zarr
import dask.array as da
from dask_ml.wrappers import ParallelPostFit
import joblib
import atexit
synchronizer = zarr.ProcessSynchronizer('.zarrsync')

def cleanup():
    if os.path.isdir('.zarrsync') and (len(os.listdir('.zarrsync')) == 0):
        shutil.rmtree('.zarrsync')
atexit.register(cleanup)

class UpscaleAge(ABC):
    
    def __init__(
            self,
            DataConfig_path: str,
            cube_config_path: str,            
            out_dir: str,
            study_name: str = 'study_name',
            study_dir: str = None,
            n_jobs: int = 1,
            n_model:int=10, 
            valid_fraction:float=0.3,
            feature_selection:bool=False,
            feature_selection_method:str=None,
            **kwargs) -> None:
        
        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)            
        self.out_dir = out_dir
        self.study_name = study_name
        
        if study_dir is None:
            study_dir = self.create_study_dir(self.out_dir, self.study_name)
            os.makedirs(study_dir, exist_ok=False)
        else:
            if not os.path.exists(study_dir):
                raise ValueError(f'restore path does not exist:\n{study_dir}')

        self.study_dir = study_dir
        self.n_jobs = n_jobs
        self.n_model= n_model
        self.valid_fraction= valid_fraction
        self.feature_selection= feature_selection
        self.feature_selection_method= feature_selection_method
     
    @staticmethod
    def _predict_func(model, 
                      input_xr,
                      chunk_size,
                      persist, 
                      proba, 
                      clean):
        x, y, = input_xr.x, input_xr.y
    
        input_data = []
    
        for var_name in input_xr.data_vars:
            input_data.append(input_xr[var_name])
    
        input_data_flattened = []
    
        for arr in input_data:
            data = arr.data.flatten().rechunk(chunk_size)
            input_data_flattened.append(data)
    
        # reshape for prediction
        input_data_flattened = da.array(input_data_flattened).transpose()
        
        # apply the classification
        out_class = model.predict(input_data_flattened)
    
        # Reshape when writing out
        out_class = out_class.reshape(len(y), len(x))
    
        # stack back into xarray
        output_xr = xr.DataArray(out_class, coords={"x": x, "y": y}, dims=["y", "x"])
    
        output_xr = output_xr.to_dataset(name="Predictions")
    
        return output_xr
     
    def model_tuning(self,
                     method:str='MLPRegressor') -> None:
        """Perform cross-validation.

        Parameters
        ----------
        n_model : int
            Number of model runs.
        valid_fraction : float
            Fraction of the validation fraction. Range between 0-1    
        feature_selection : bool
            Whether to do feature selection.
        feature_selection_method : str
            The method to use for the feature selections
        """
        
        cluster_ = xr.open_dataset(self.DataConfig['training_dataset']).cluster.values
        np.random.shuffle(cluster_)
        
        train_subset, valid_subset = train_test_split(cluster_, test_size=self.DataConfig['valid_fraction'], shuffle=True)
        self.DataConfig['method'] = method
        mlp_method = MLPmethod(tune_dir=os.path.join(self.study_dir, "tune"), 
                               DataConfig= self.DataConfig, 
                               method = method)
        mlp_method.train(train_subset=train_subset,
                          valid_subset=valid_subset,
                          feature_selection= self.DataConfig['feature_selection'],
                          feature_selection_method=self.DataConfig['feature_selection_method'],
                          n_jobs = self.n_jobs)
        if not os.path.exists(self.study_dir + '/save_model/'):
            os.makedirs(self.study_dir + '/save_model/')
            
        shutil.rmtree(os.path.join(self.study_dir, "tune"))        
        return mlp_method.best_model
    
        def predict_xr(self,
                       model,
                       input_xr,
                       chunk_size=None,
                       persist=False,
                       proba=False,
                       clean=True,
                       return_input=False):
            """
            Using dask-ml ParallelPostfit(), runs  the parallel
            predict and predict_proba methods of sklearn
            estimators. Useful for running predictions
            on a larger-than-RAM datasets.
            Last modified: September 2020
            Parameters
            ----------
            model : scikit-learn model or compatible object
                Must have a .predict() method that takes numpy arrays.
            input_xr : xarray.DataArray or xarray.Dataset.
                Must have dimensions 'x' and 'y'
            chunk_size : int
                The dask chunk size to use on the flattened array. If this
                is left as None, then the chunks size is inferred from the
                .chunks method on the `input_xr`
            persist : bool
                If True, and proba=True, then 'input_xr' data will be
                loaded into distributed memory. This will ensure data
                is not loaded twice for the prediction of probabilities,
                but this will only work if the data is not larger than
                distributed RAM.
            proba : bool
                If True, predict probabilities
            clean : bool
                If True, remove Infs and NaNs from input and output arrays
            Returns
            ----------
            output_xr : xarray.Dataset
                An xarray.Dataset containing the prediction output from model.
                if proba=True then dataset will also contain probabilites, and
                Has the same spatiotemporal structure as input_xr.
            """
            model = ParallelPostFit(model)
            with joblib.parallel_backend("dask", wait_for_workers_timeout=20):
                output_xr = self._predict_func(
                                                model, input_xr, persist
                                                )       
    
            return output_xr

    def ForwardRun(self,
                n_model:int,
                MLRegressor_path:str=None,
                MLPClassifier_path:str=None):
        
        for run_ in tqdm(np.arange(n_model), desc='Forward run model members'):
            
            cube = DataCube(cube_config = self.cube_config)
            
            best_regressor  = self.model_tuning(method = 'MLPRegressor')
            best_classifier = self.model_tuning(method = 'MLPClassifier')
            
            X_upscale_class = zarr.open(self.DataConfig, synchronizer=synchronizer)[self.DataConfig['features']]
            X_upscale_reg = zarr.open(self.DataConfig, synchronizer=synchronizer)[self.DataConfig['features']]
            OrigShape = X_upscale_class.shape            
            RF_pred = np.zeros(X_upscale_class.shape[0]) * np.nan
            mask = (np.all(np.isfinite(X_upscale_class), axis=1)) & (np.all(np.isfinite(X_upscale_reg), axis=1))
            
            if (X_upscale_class[mask].shape[0]>0):
                pred_ = best_classifier.predict_(X_upscale_class[mask])
                pred_[pred_==1] = 300
                pred_reg= best_regressor.predict_(X_upscale_reg[mask])
                pred_reg[pred_reg>=300] = 299
                pred_reg[pred_reg<0] = 0                
                pred_[pred_==0] = pred_reg[pred_==0]
                RF_pred[mask] = pred_    
    
            RF_pred = RF_pred.reshape(OrigShape)
            cube.update(RF_pred)
