from __future__ import annotations
import os
import xarray as xr
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import yaml as yml
import shutil
from ageUpscaling.utils.utilities import TimeKeeper
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.cube.cube import DataCube

class Experiment(object):
    """Experiment class used for HP tuning, cross validation, model training, prediction.

    Usage
    -----
    An experiment has four main components, each subclasses `ExperimentModule`:

    method (subclass of `BaseMethod`)
        A model implemented in the BaseMethod framework
    provider (subclass of `BaseProvider`)
        A data provider
    DataConfig (instance of DataConfig)custom_exp_name
        A data configuration

    Directory structure
    -------------------
    * The experiment directory (exp_dir): <base_dir>/<exp_name>/<version_XX>
    * The version is created automatically, override `.create_experiment_dir(...)` for custom structure.
    * If an exp_dir is passed, the experiment is restored.

    Parameters
    ----------
    method : BaseMethod
        A method subclassing BaseMethod.
    provider : BaseProvider
        A provider subclassing BaseProvider.
    site_splitter : BaseSplitter
        A site splitter subclassing BaseSplitter.
    DataConfig : DataConfig
        A data configuration.
    hp_params : hp_search_space
        A hyper-parameter space.
        
    base_dir : str
        The experiment base directory. Default is '/Net/Groups/BGI/scratch/splcClassifier/experiments'.
        See `directory structure` for further details.
    exp_name : str = 'exp_name'
        The experiment name.
        See `directory structure` for further details.
    exp_dir : Optional[str] = None
        The restore directory. If passed, an existing experiment is loaded.
        See `directory structure` for further details.
    n_jobs : int = 1
        Number of workers, is used in various places. TODO: specify.
    training_mask : Optional[str]
        An optional training mask. TODO: more details.
    **kwargs:
        Additional keyword arguments are passed to `method`. TODO: more details.

    """
    def __init__(
            self,
            DataConfig_path: str,
            base_dir: str,
            exp_name: str = 'exp_name',
            exp_dir: str = None,
            n_jobs: int = 1,
            n_trials:int = 2,
            **kwargs):

        with open(DataConfig_path, 'r') as f:
            self.DataConfig =  yml.safe_load(f)
        self.base_dir = base_dir
        self.exp_name = exp_name
        
        if exp_dir is None:
            exp_dir = self.create_experiment_dir(self.base_dir, self.exp_name)
            os.makedirs(exp_dir, exist_ok=False)
        else:
            if not os.path.exists(exp_dir):
                raise ValueError(f'restore path does not exist:\n{exp_dir}')

        self.exp_dir = exp_dir
        self.n_jobs = n_jobs
        self.n_trials = n_trials

    def create_experiment_dir(self, base_dir: str, exp_name: str) -> str:
        """Create experiment directory

        Parameter
        ---------
        base_dir : str
            The base directory.
        exp_name : str
            The experiment name.

        Returns
        -------
        <base_dir>//<exp_name>/<v_0000>
        """
        exp_dir = os.path.join(base_dir, exp_name)
        exp_dir = self.next_version_path(exp_dir, prefix='v_')
        return exp_dir
    
    @staticmethod
    def create_and_get_path(*loc, exist_ok=True, is_file_path=False):
        if len(loc) > 0:
            path = os.path.join(*loc)
        else:
            path = ''

        if is_file_path:
            create_path = os.path.dirname(path)
        else:
            create_path = path

        if not os.path.exists(create_path):
            os.makedirs(create_path, exist_ok=exist_ok)
        return path

    @staticmethod
    def next_version_path(
            dir_path: str,
            prefix: str = 'v',
            postfix: str = '',
            num_digits: int = 4,
            create: bool = True) -> str:
        """
        Finds the next incremental path from a pattern.

        Create pattern `my/dir/[prefix][DD][postfix]`, `DD` being an incremental version number:


        Example:

        Exp.next_version_path('my/dir', prefix='version-', postfix='.json', num_digits=2)

        my/dir/version-00.json <- exists
        my/dir/version-05.json <- exists
        my/dir/version-06.json <- doesn't exist, is returned

        Note that the largest matching version number is used even if gaps exist, i.e., if
        version `00` and `05` exist, the returned version is `06`.

        Parameters
        ----------
        dir_path:
            the base directory. Must exist if `create=False`.
        prefix:
            the version string prefix, default is `v`.
        postfix:
            the version string postfix, e.g., `.json`.
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
            if f.startswith(prefix) and f.endswith(postfix):
                f_len = len(f)
                version_string = f[len(prefix):f_len - len(postfix)]
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

        return f'{os.path.join(os.path.join(dir_path, prefix))}{version_digits}{postfix}'

    @property
    def tune_dir(self):
        return self.create_and_get_path(self.exp_dir, 'tune')
    
    @property
    def pred_dir(self):
        return self.create_and_get_path(self.exp_dir, 'save_pred')
    
    def xval(self, 
             n_folds:int=10, 
             valid_fraction:float=0.3,
             feature_selection:bool=False,
             prediction:bool=True) -> None:
        """Perform cross-validation.

        Parameters
        ----------
        n_folds : int
            Number of cross-validation folds.
        valid_fraction : float
            Fraction of the validation fraction. Range between 0-1    
        predict : bool
            Whether to predict test set. Default is `True`.
        predict_train : bool
            Whether to predict training set. Default is `True`.
            
        kwargs :
            Are passe to `self.train(...)`
        """
        
        cluster_ = xr.open_dataset(self.DataConfig['cube_path']).cluster.values
        sample_ = xr.open_dataset(self.DataConfig['cube_path']).sample.values
        pred_cube = DataCube(os.path.join(self.pred_dir, "model_pred"),
                             njobs=1,
                             coords={'cluster': cluster_,
                                     'sample': sample_})
        np.random.shuffle(cluster_)
        kf = KFold(n_splits=n_folds)
        timekeeper = TimeKeeper(n_folds=n_folds)
        for train_index, test_index in kf.split(cluster_):
            train_subset, test_subset = cluster_[train_index], cluster_[test_index]
            train_subset, valid_subset = train_test_split(train_subset, test_size=valid_fraction, shuffle=True)
            mlp_method = MLPmethod(tune_dir=self.tune_dir, DataConfig= self.DataConfig)
            mlp_method.train(train_subset=train_subset,
                              valid_subset=valid_subset, 
                              test_subset=test_subset, 
                              feature_selection= feature_selection)
            if prediction:
                mlp_method.predict_xr(save_cube = pred_cube)                       
            timekeeper.lap(message="Time to run fold: {lap_time}")
            timekeeper.time_left(message="Total time: {total_time}, est. remaining: {time_left}")
            print('=' * 20)
            shutil.rmtree(self.tune_dir)
            
    
    
    
    

    
