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
            out_dir: str,
            study_name: str = 'study_name',
            study_dir: str = None,
            n_jobs: int = 1,
            **kwargs):

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
        self.cube_config['cube_location'] = os.path.join(study_dir, 'model_output')
        self.n_jobs = n_jobs

    def create_study_dir(self, out_dir: str, study_name: str) -> str:
        """Create study directory

        Parameter
        ---------
        out_dir : str
            The base directory.
        study_name : str
            The studyt name.

        Returns
        -------
        <out_dir>//<study_name>/<v_0000>
        """
        study_dir = os.path.join(out_dir, study_name)
        study_dir = self.next_version_path(study_dir, prefix='v_')
        return study_dir
    
    @staticmethod
    def next_version_path(
            dir_path: str,
            prefix: str = 'v',
            num_digits: int = 4,
            create: bool = True) -> str:
        """
        Finds the next incremental path from a pattern.

        Create pattern `my/dir/[prefix][DD]`, `DD` being an incremental version number:


        Example:

        Exp.next_version_path('my/dir', prefix='version-', num_digits=2)

        my/dir/version-00 <- exists
        my/dir/version-05 <- exists
        my/dir/version-06 <- doesn't exist, is returned

        Note that the largest matching version number is used even if gaps exist, i.e., if
        version `00` and `05` exist, the returned version is `06`.

        Parameters
        ----------
        dir_path:
            the base directory. Must exist if `create=False`.
        prefix:
            the version string prefix, default is `v`.
        num_digits:
            The number of digits to use in incremental version numbering.
        create:
            Create `dir_path` if the it does not exist.

        Returns
        -------
        The version path (str).
        """

        if num_digits < 1:
            raise ValueError(
                f'`num_digits` must be an integer > 0, is {num_digits}.'
            )

        if not os.path.exists(dir_path):
            if create:
                os.makedirs(dir_path)
            else:
                raise FileNotFoundError(
                    'No such file or directory: `{dir_path}`. Use `create=True`?'
                )

        max_v = -1

        for f in os.listdir(dir_path):
            if f.startswith(prefix):
                f_len = len(f)
                version_string = f[len(prefix):f_len]
                if len(version_string) == num_digits:
                    try:
                        version_int = int(version_string)
                    except ValueError as e:
                        continue

                    max_v = max(max_v, version_int)

        version_nr = max_v + 1

        version_range = 10**num_digits - 1
        if version_nr > version_range:
            raise ValueError(
                f'The next incremental version is `{version_nr}`, which is out of range  (1, {version_range}) '
                f'given `num_digits={num_digits}`. Use `prefix="{prefix}new_test2"` to continue ;)'
            )

        pattern = f'%0{num_digits}d'
        version_digits = pattern % version_nr

        return f'{os.path.join(os.path.join(dir_path, prefix))}{version_digits}'
    
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
            
    
            
    
    
    
    

    
