#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   study.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for cross validation, model training, prediction
"""
import os

import numpy as np
import xarray as xr
from abc import ABC
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import yaml as yml
import shutil

from ageUpscaling.core.cube import DataCube
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.methods.xgboost import XGBoost
from ageUpscaling.methods.RandomForest import RandomForest
from ageUpscaling.methods.autoML import TPOT

class Study(ABC):
    """Study abstract class for cross validation, model training, prediction.

    Parameters
    ----------
    DataConfig_path : str
        Path to the data configuration file.     
    cube_config_path : str
        Path to the cube configuration file.
    base_dir : str
        The base directory for the study. See `directory structure` for further details.
    algorithm : str, optional
        The algorithm name. Default is 'MLP'.
        See `directory structure` for further details.
    study_dir : str, optional
        The directory to restore an existing study. If passed, an existing study is loaded.
        See `directory structure` for further details.
    n_jobs : int, optional
        Number of workers. Default is 1.
    **kwargs : additional keyword arguments
        Additional keyword arguments.

    """
    def __init__(self,
                 DataConfig_path: str,
                 cube_config_path: str,            
                 base_dir: str,
                 algorithm: str = 'MLP',
                 exp_name: str = None,
                 study_dir: str = None,
                 n_jobs: int = 1,
                 **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
        
        self.base_dir = base_dir
        self.algorithm = algorithm
        self.exp_name = exp_name
        
        if study_dir is None:
            study_dir = self.version_dir(self.base_dir, self.exp_name, self.algorithm)
            os.makedirs(study_dir, exist_ok=False)
        else:
            if not os.path.exists(study_dir):
                raise ValueError(f'restore path does not exist:\n{study_dir}')

        self.study_dir = study_dir
        self.cube_config['cube_location'] = os.path.join(study_dir, 'model_output')
        self.n_jobs = n_jobs
    
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
    
    def cross_validation(self, 
                         n_folds:int=10, 
                         valid_fraction:float=0.3,
                         feature_selection:bool=False,
                         feature_selection_method:str=None) -> None:
        """Perform cross-validation on the data.
    
        Parameters
        ----------
        n_folds : int, optional
            The number of cross-validation folds.
            Default is 10.
        valid_fraction : float, optional
            The fraction of the data to use as the validation set.
            Range is between 0 and 1.
            Default is 0.3.
        feature_selection : bool, optional
            Whether to perform feature selection before training the model.
            Default is False.
        feature_selection_method : str, optional
            The method to use for feature selection.
            Only applicable if `feature_selection` is True.
            Default is None.
    
        Notes
        -----
        - If `feature_selection` is True, `feature_selection_method` must be specified.
        """
        
        pred_cube = DataCube(cube_config = self.cube_config)
        cluster_ = xr.open_dataset(self.DataConfig['training_dataset']).cluster.values
        np.random.shuffle(cluster_)
        kf = KFold(n_splits=n_folds)
        
        for train_index, test_index in tqdm( kf.split(cluster_), desc='Performing cross-validation'):
            train_subset, test_subset = cluster_[train_index], cluster_[test_index]
            train_subset, valid_subset = train_test_split(train_subset, test_size=valid_fraction, shuffle=True)
            
            for task_ in ["Regressor", "Classifier"]:
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
                                  test_subset=test_subset, 
                                  feature_selection= feature_selection,
                                  feature_selection_method=feature_selection_method,
                                  n_jobs = self.n_jobs)
                ml_method.predict_clusters(save_cube = pred_cube)                       
                shutil.rmtree(os.path.join(self.study_dir, "tune"))
            
    
            
    
    
    
    

    
