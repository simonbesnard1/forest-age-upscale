from __future__ import annotations
import os
import xarray as xr
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import yaml as yml
import shutil
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.core.cube import DataCube
from abc import ABC
from tqdm import tqdm

class Study(ABC):
    """Study abstract class used for cross validation, model training, prediction.

    Parameters
    ----------
    DataConfig_path : DataConfig_path
        A data configuration path.     
    out_dir : str
        The study base directory.
        See `directory structure` for further details.
    study_name : str = 'study_name'
        The study name.
        See `directory structure` for further details.
    study_dir : Optional[str] = None
        The restore directory. If passed, an existing study is loaded.
        See `directory structure` for further details.
    n_jobs : int = 1
        Number of workers.

    """
    def __init__(
            self,
            DataConfig_path: str,
            cube_config_path: str,            
            base_dir: str,
            study_name: str = 'study_name',
            study_dir: str = None,
            n_jobs: int = 1,
            **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        
        with open(cube_config_path, 'r') as f:
            self.cube_config =  yml.safe_load(f)
        
        self.base_dir = base_dir
        self.study_name = study_name
        
        if study_dir is None:
            study_dir = self.version_dir(self.base_dir, self.study_name)
            os.makedirs(study_dir, exist_ok=False)
        else:
            if not os.path.exists(study_dir):
                raise ValueError(f'restore path does not exist:\n{study_dir}')

        self.study_dir = study_dir
        self.cube_config['cube_location'] = os.path.join(study_dir, 'model_output')
        self.n_jobs = n_jobs
    
    def version_dir(self, 
                    base_dir: str, 
                    study_name: str) -> str:
        """
        Creates a new version of a directory by appending the version number to the end of the directory name.
        If the directory already exists, it will be renamed to include the version number before the new directory is created.
        """
        
        # Return the name of the new directory
        return self.increment_dir_version(base_dir, study_name)
    
    @staticmethod
    def increment_dir_version(base_dir: str,
                              study_name:str) -> str:
        """
        Increments the version of a directory by appending the next available version number to the end of the directory name.
        """
        # Get a list of all directories that start with the given directory name
        dir_list = [d for d in os.listdir(base_dir) if d.startswith(study_name)]
        
        # Sort the list of directories in ascending order
        dir_list.sort()
        
        # Check if the list of directories is empty
        if len(dir_list) == 0:
            # If the list is empty, this is the first version of the directory
            # Set the version number to "1.0"
            version = "1.0"
        else:
            # If the list is not empty, get the last directory in the list
            # This will be the most recent version of the directory
            last_dir = dir_list[-1]
            
            # Split the directory name into its base name and version number
            study_name, version = last_dir.split("-")
            
            # Increment the version number
            major, minor = version.split(".")
            major = int(major)
            minor = int(minor)
            minor += 1
            if minor >= 10:
                major += 1
                minor = 0
            version = f"{major}.{minor}"
        
        # Return the name of the new directory
        return f"{base_dir}/{study_name}-{version}"
    
    def cross_validation(self, 
                         method:str='MLPRegressor',
                         n_folds:int=10, 
                         valid_fraction:float=0.3,
                         feature_selection:bool=False,
                         feature_selection_method:str=None) -> None:
        """Perform cross-validation.

        Parameters
        ----------
        n_folds : int
            Number of cross-validation folds.
        valid_fraction : float
            Fraction of the validation fraction. Range between 0-1    
        feature_selection : bool
            Whether to do feature selection.
        feature_selection_method : str
            The method to use for the feature selections
        """
        
        pred_cube = DataCube(cube_config = self.cube_config)
        cluster_ = xr.open_dataset(self.DataConfig['training_dataset']).cluster.values
        np.random.shuffle(cluster_)
        kf = KFold(n_splits=n_folds)
        
        for train_index, test_index in tqdm( kf.split(cluster_), desc='Performing cross-validation'):
            train_subset, test_subset = cluster_[train_index], cluster_[test_index]
            train_subset, valid_subset = train_test_split(train_subset, test_size=valid_fraction, shuffle=True)
            mlp_method = MLPmethod(tune_dir=os.path.join(self.study_dir, "tune"), 
                                   DataConfig= self.DataConfig,
                                   method=method)
            mlp_method.train(train_subset=train_subset,
                              valid_subset=valid_subset, 
                              test_subset=test_subset, 
                              feature_selection= feature_selection,
                              feature_selection_method=feature_selection_method,
                              n_jobs = self.n_jobs)
            mlp_method.predict_clusters(save_cube = pred_cube)                       
            shutil.rmtree(os.path.join(self.study_dir, "tune"))
            
    
            
    
    
    
    

    
