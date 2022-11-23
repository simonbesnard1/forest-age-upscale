from __future__ import annotations
import os
from numpy.core import numeric
import xarray as xr
import numpy as np
from typing import Optional, Dict, Any, Union
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from splcClassifier.core.experiment_utils import ExperimentModule
from splcClassifier.core.data_config import DataConfig
from splcClassifier.providers.base import BaseProvider
from splcClassifier.methods.base import BaseMethod
from splcClassifier.core.experiment_utils import TimeKeeper
from splcClassifier.methods.MLmethods import MLmodel
from splcClassifier.methods.DLmethods import DLmodel
from splcClassifier.core.hpo_utils import hp_search_space

class Experiment(ExperimentModule):
    """Experiment class used for HP tuning, cross validation, model training, prediction.

    Usage
    -----
    An experiment has four main components, each subclasses `ExperimentModule`:

    method (subclass of `BaseMethod`)
        A model implemented in the BaseMethod framework
    provider (subclass of `BaseProvider`)
        A data provider
    data_config (instance of DataConfig)custom_exp_name
        A data configuration

    Directory structure
    -------------------
    * The experiment directory (exp_dir): <base_dir>/<exp_group>/<exp_name>/<version_XX>
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
    data_config : DataConfig
        A data configuration.
    hp_params : hp_search_space
        A hyper-parameter space.
        
    base_dir : str
        The experiment base directory. Default is '/Net/Groups/BGI/scratch/splcClassifier/experiments'.
        See `directory structure` for further details.
    exp_group : Optional[str]
        An experiment group. If not passed, the group will be the user name (e.g. `root`).
        See `directory structure` for further details.
    exp_name : str = 'exp_name'
        The experiment name.
        See `directory structure` for further details.
    exp_desc : Optional[str]
        An experiment description.
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
            model: str,
            provider: BaseProvider,
            data_config: DataConfig,
            hp_params: hp_search_space,
            base_dir: str,
            exp_group: Optional[str] = None,
            exp_name: str = 'exp_name',
            exp_desc: str = 'no description',
            exp_dir: Optional[str] = None,
            n_jobs: int = 1,
            n_trials:int = 2,
            **kwargs):

        self.model = model
        self.provider = provider
        self.data_config = data_config
        self._hparams = hp_params
        self.base_dir = base_dir
        self.exp_group = exp_group
        self.exp_name = exp_name
        self.exp_desc = 'no description' if exp_desc is None else exp_desc
        
        if self.model == 'XGBoost':
            self.method = MLmodel(provider = self.provider,  data_config = self.data_config, model = self.model)
        elif self.model == 'UNET' or self.model == 'DeepLab':
            self.method = DLmodel(provider = self.provider,  data_config = self.data_config, model = self.model)
        
        if exp_dir is None:
            exp_dir = self.create_experiment_dir(self.base_dir, self.exp_group, self.exp_name)
            os.makedirs(exp_dir, exist_ok=False)
        else:
            if not os.path.exists(exp_dir):
                raise ValueError(f'restore path does not exist:\n{exp_dir}')

        self.exp_dir = exp_dir
        self.n_jobs = n_jobs
        self.n_trials = n_trials

    def create_experiment_dir(self, base_dir: str, exp_group: str, exp_name: str) -> str:
        """Create experiment directory

        Parameter
        ---------
        base_dir : str
            The base directory.
        exp_group : str
            The experiment group.
        exp_name : str
            The experiment name.

        Returns
        -------
        <base_dir>/<exp_group>/<exp_name>/<v_0000>
        """
        exp_dir = os.path.join(base_dir, exp_group, exp_name)
        exp_dir = self.next_version_path(exp_dir, prefix='v_')
        return exp_dir

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
    def xval_dir(self):
        return self.create_and_get_path(self.exp_dir, 'xval')
    def predict(
            self,
            model: Optional[Union[BaseMethod, str]] = None,
            # best_trial:int = None, 
            subset_chuncks: np.array = {},
            **kwargs) -> xr.Dataset:
        """Make prediction on a trained mopdel.

        Parameters
        ----------
        model : BaseMethod or str
            The trained model. 
        subset : dict
            An optinal subset to be applied befor making the predictions, e.g., dict(time=slice('2001', '2004')).
        **kwargs :
            Are passed to `model.predict(...)`

        Returns
        -------
        The predictions and labels, an xr.Dataset.
        """
        if model is None:
            model = self.final_model
        elif type(model) is str:
            model = self.method.load(*os.path.split(model))
        else:
            pass
        
        _target, _pred = self.method.predict(model,
                                             subset_chuncks=subset_chuncks)
        return xr.merge([_target, _pred])

    def model_tune(
            self,
            fold,
            train_chuncks: np.array = {},
            valid_chuncks: np.array = {},
            hparam_sample:Dict[str, Any]= {},
            exp_name:str = {},
            exp_dir:str = {},
            tune_dir:str = {},
            n_jobs:int = 1,
            n_trials:int=1) -> numeric:
        """Model training.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.
        train_subset : dict
            data subset for the trining set.
        valid_subset : dict
            data subset for the validation set.
        kwargs :
            Additional hyper-parameters passed to `method`.

        Returns
        -------
        loss: the validation loss after model training.
        """
        
        # best_model, best_trial = self.method.train(fold = fold,
        best_model = self.method.train(fold = fold,
                                       train_chuncks = train_chuncks,
                                       valid_chuncks = valid_chuncks,
                                       hparam_sample=hparam_sample,
                                       exp_name = exp_name,
                                       exp_dir = exp_dir,
                                       tune_dir = tune_dir,
                                       n_jobs = n_jobs,
                                       n_trials = n_trials)
        return best_model #, best_trial
    
    def xval(self, 
             n_folds:int, 
             valid_fraction:float, 
             predict: bool, 
             predict_train:bool) -> None:
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
        
        #folds_ = np.array(list(set([f for f in os.listdir(self.provider.path_cube + '/chunck') if not f.startswith('.')]))).astype(int)
        folds_ = xr.open_zarr(self.provider.path_cube).chunck.values
        np.random.shuffle(folds_)
        kf = KFold(n_splits=n_folds)
        iter_ = 1
        timekeeper = TimeKeeper(n_folds=n_folds)
        for train_index, test_index in kf.split(folds_):
            #print(f'\nRunning fold{iter_}')
            training_chuncks, testing_chuncks = folds_[train_index], folds_[test_index]
            training_chuncks, valid_chuncks = train_test_split(training_chuncks, test_size=valid_fraction, shuffle=True)
            
            if not os.path.exists(self.tune_dir + '/save_model/fold' + str(iter_)):
                os.makedirs(self.tune_dir + '/save_model/fold' + str(iter_))
            
            self.final_model = self.model_tune(fold = iter_, # , best_trial
                                                train_chuncks = training_chuncks,
                                                valid_chuncks = valid_chuncks,
                                                hparam_sample=self._hparams,
                                                exp_name = self.exp_name,
                                                exp_dir = self.exp_dir,
                                                tune_dir=self.tune_dir,
                                                n_jobs = self.n_jobs,
                                                n_trials = self.n_trials)
            # print(self.final_model)
            # print(self.final_model['trainer'])
            # print(self.final_model[1]['model'])
            # print(self.final_model[0]['trainer'])
            # print(self.final_model['model'])
            # print(self.final_model[1])


            if predict:
                out_ontest = self.predict(model=self.final_model, subset_chuncks= testing_chuncks)
                out_ontest.to_zarr(self.xval_dir + '/fold' + str(iter_) + '/test/')
            
            if predict_train:
                out_ontrain = self.predict(model=self.final_model, subset_chuncks= training_chuncks)
                out_ontrain.to_zarr(self.xval_dir + '/fold' + str(iter_) + '/train/')
            
            timekeeper.lap(message="Time to run fold: {lap_time}")
            timekeeper.time_left(message="Total time: {total_time}, est. remaining: {time_left}")
            iter_ += 1
            print('=' * 20)
    
    
    
    

    
