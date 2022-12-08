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
import pickle
from sklearn.model_selection import train_test_split
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.core.cube import DataCube
import shutil
import zarr

class UpscaleAge(ABC):
    
    def __init__(
            self,
            DataConfig_path: str,
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
        self.out_dir = out_dir
        self.study_name = study_name
        
        if study_dir is None:
            study_dir = self.create_study_dir(self.out_dir, self.study_name)
            os.makedirs(study_dir, exist_ok=False)
        else:
            if not os.path.exists(study_dir):
                raise ValueError(f'restore path does not exist:\n{study_dir}')

        self.study_dir = study_dir
        self.DataConfig['cube_location'] = os.path.join(study_dir, 'model_output')
        self.n_jobs = n_jobs
        self.n_model= n_model
        self.valid_fraction= valid_fraction
        self.feature_selection= feature_selection
        self.feature_selection_method= feature_selection_method
        self.feature_selection_method= feature_selection_method
        
    def model_tuning(self, 
                     valid_fraction:float=0.3,
                     feature_selection:bool=False,
                     feature_selection_method:str=None) -> None:
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
        
        train_subset, valid_subset = train_test_split(cluster_, test_size=valid_fraction, shuffle=True)
        mlp_method = MLPmethod(tune_dir=os.path.join(self.study_dir, "tune"), DataConfig= self.DataConfig)
        mlp_method.train(train_subset=train_subset,
                          valid_subset=valid_subset,
                          feature_selection= feature_selection,
                          feature_selection_method=feature_selection_method,
                          n_jobs = self.n_jobs)
        if not os.path.exists(self.study_dir + '/save_model/'):
            os.makedirs(self.study_dir + '/save_model/')
            
        shutil.rmtree(os.path.join(self.study_dir, "tune"))        
        return mlp_method.best_model

    def Forward(self,
                n_model:int,
                retrain:bool= True,
                MLRegressor_path:str=None,
                MLPClassifier_path:str=None):
        
        for run_ in tqdm(np.arange(n_model), desc='Forward run model members'):
            
            cube = DataCube()
            if retrain:
                best_regressor  = self.model_tuning(method = 'MLPregressor')
                best_classifier = self.model_tuning(method = 'MLPclassifier')
            elif not retrain:
                best_regressor  = pickle.load(MLRegressor_path)
                best_classifier = pickle.load(MLPClassifier_path)
            
            X_upscale_class = zarr.open()
            X_upscale_reg = zarr.open()
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
