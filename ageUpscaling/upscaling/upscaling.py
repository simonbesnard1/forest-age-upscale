from __future__ import annotations
import os
import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
import yaml as yml
import shutil
from ageUpscaling.methods.MLP import MLPmethod
from ageUpscaling.core.cube import DataCube
from abc import ABC
from tqdm import tqdm
import joblib
import atexit
import zarr
from dask_ml.wrappers import ParallelPostFit
import dask.array as da
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
        self.n_jobs = n_jobs
        self.valid_fraction= self.DataConfig["valid_fraction"]
        self.feature_selection= self.DataConfig["feature_selection"]
        self.feature_selection_method= self.DataConfig["feature_selection_method"]           
    
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
    
    def _predict_func(self, 
                      model, 
                      input_xr):
        x, y, chunk_size, = input_xr.longitude, input_xr.latitude, input_xr.chunk_size
    
        input_data = []
    
        for var_name in input_xr.data_vars:
            input_data.append(self.norm(input_xr[var_name], model['norm_stats'][var_name]))
    
        input_data_flattened = []
    
        for arr in input_data:
            data = arr.data.flatten().rechunk(chunk_size)
            input_data_flattened.append(data)
    
        input_data_flattened = da.array(input_data_flattened).transpose()
        
        input_data_flattened = da.where(
                da.isfinite(input_data_flattened), input_data_flattened, 0)
        
        out_ = self.denorm_target(model['best_model'].predict(input_data_flattened), 
                                  model['norm_stats']['age'])
        out_ = da.where(da.isfinite(out_), out_, 0)
    
        out_ = out_.reshape(len(y), len(x))
    
        output_xr = xr.DataArray(out_, coords={"longitude": x, "latitude": y}, dims=["longitude", "latitude"])
    
        output_xr = output_xr.to_dataset(name="forest_age")
    
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
        return {'best_model': mlp_method.best_model, 'selected_features': mlp_method.final_features, 'norm_stats' : mlp_method.mldata.norm_stats}
    
    def predict_xr(self,
                   model,
                   input_xr):
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
                                            model, input_xr
                                            )       

        return output_xr
    
    def ForwardRun(self,
                   tree_cover_treshold:int = 10,
                   MLRegressor_path:str=None,
                   MLPClassifier_path:str=None):
        
        for run_ in tqdm(np.arange(self.cube_config['output_writer_params']['dims']['members']), desc='Forward run model members'):
            
            cube = DataCube(cube_config = self.cube_config)
            
            best_regressor  = self.model_tuning(method = 'MLPRegressor')
            best_classifier = self.model_tuning(method = 'MLPClassifier')
            
            X_upscale_class = xr.open_zarr(self.DataConfig['global_cube'], 
                                           synchronizer=synchronizer)[best_classifier['selected_features']]
            X_upscale_reg = xr.open_zarr(self.DataConfig['global_cube'], 
                                         synchronizer=synchronizer)[best_regressor['selected_features']]
            
            out_ = self.predict_xr(best_regressor, X_upscale_reg)
            
            # OrigShape = X_upscale_class.shape            
            # RF_pred = np.zeros(X_upscale_class.shape[0]) * np.nan
            # mask = (np.all(np.isfinite(X_upscale_class), axis=1)) & (np.all(np.isfinite(X_upscale_reg), axis=1))
            
            # if (X_upscale_class[mask].shape[0]>0):
            #     pred_ = best_classifier.predict_(X_upscale_class[mask])
            #     pred_[pred_==1] = 300
            #     pred_reg= best_regressor.predict_(X_upscale_reg[mask])
            #     pred_reg[pred_reg>=300] = 299
            #     pred_reg[pred_reg<0] = 0                
            #     pred_[pred_==0] = pred_reg[pred_==0]
            #     RF_pred[mask] = pred_    
    
            # RF_pred = RF_pred.reshape(OrigShape)
            cube.update(out_)
            
    def norm(self, 
             x: np.array,
             norm_stats:dict) -> np.array:
        """Returns de-normalize target, last dimension of `x` must match len of `self.target_norm_stats`."""
        
        return (x - norm_stats['mean']) / norm_stats['std'] 
    
    def denorm_target(self, 
                      x: np.array,
                      norm_stats:dict) -> np.array:
        """Returns de-normalize target, last dimension of `x` must match len of `self.target_norm_stats`."""
        
        return x * norm_stats['std'] + norm_stats['mean']
    
    
    
    

    
