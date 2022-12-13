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
import atexit
import zarr
import dask.array as da
import multiprocessing as mp
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
                      params) -> None:
        
        subset_cube = params['feature_cube'].sel(latitude= params['latitude'],longitude=params['longitude'])
        
        X_upscale_class = []
        for var_name in params["best_classifier"]['selected_features']:
            X_upscale_class.append(self.norm(subset_cube[var_name], params["best_classifier"]['norm_stats'][var_name]))
            
        X_upscale_reg = []
        for var_name in params["best_regressor"]['selected_features']:
            X_upscale_reg.append(self.norm(subset_cube[var_name], params["best_regressor"]['norm_stats'][var_name]))
        
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
        
        RF_pred = np.zeros(X_upscale_reg_flattened.shape[0]) * np.nan
        mask = (np.all(np.isfinite(X_upscale_reg_flattened), axis=1)) & (np.all(np.isfinite(X_upscale_class_flattened), axis=1))
        
        if (X_upscale_class_flattened[mask].shape[0]>0):
            pred_ = params["best_classifier"]['best_model'].predict(X_upscale_class_flattened[mask])
            pred_[pred_==1] = params["max_forest_age"][0]
            pred_reg= self.denorm_target(params["best_regressor"]['best_model'].predict(X_upscale_reg_flattened[mask]), 
                                         params["best_regressor"]['norm_stats']['age'])
            pred_reg[pred_reg>=params["max_forest_age"][0]] = params["max_forest_age"][0] -1
            pred_reg[pred_reg<0] = 0                
            pred_[pred_== 0] = pred_reg[pred_==0]
            RF_pred[mask] = pred_
            out_ = RF_pred.reshape(len(subset_cube.latitude), len(subset_cube.longitude), 1)
            output_xr = xr.DataArray(out_, coords={"latitude": subset_cube.latitude, 
                                                   "longitude": subset_cube.longitude, 
                                                   'members': [params["member"]]}, dims=["latitude", "longitude", "members"])
            output_xr = output_xr.to_dataset(name="forest_age")
            params['pred_cube'].compute_cube(output_xr)
        
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
    
    def ForwardRun(self,
                   tree_cover_treshold:int = '010',
                   MLRegressor_path:str=None,
                   MLPClassifier_path:str=None,
                   nLatChunks:int=50,
                   nLonChunks:int=2,
                   njobs:int = 10):
        
        for run_ in tqdm(np.arange(self.cube_config['output_writer_params']['dims']['members']), desc='Forward run model members'):
            
            pred_cube           = DataCube(cube_config = self.cube_config)
            feature_cube        = xr.open_zarr(self.DataConfig['global_cube'], synchronizer=synchronizer)
            feature_cube        = feature_cube.rename({'agb_001deg_cc_min_{tree_cover}'.format(tree_cover = tree_cover_treshold) : 'agb'})
            
            best_regressor      = self.model_tuning(method = 'MLPRegressor')
            best_classifier     = self.model_tuning(method = 'MLPClassifier')
            
            LatChunks           = np.linspace(90,-90,nLatChunks)
            LonChunks           = np.linspace(-180,180,nLonChunks)
            AllExtents          = []
            for lat in range(nLatChunks-1):
                for lon in range(nLonChunks-1):
                    AllExtents.append({'latitude':slice(LatChunks[lat],LatChunks[lat+1]),
                                       'longitude':slice(LonChunks[lon],LonChunks[lon+1]),
                                        'best_regressor': best_regressor,
                                        'best_classifier': best_classifier,
                                        'feature_cube': feature_cube,
                                        'pred_cube': pred_cube,
                                        'member':run_,
                                        'max_forest_age': self.DataConfig['max_forest_age']})
                  
            p=mp.Pool(njobs,maxtasksperchild=1)
            p.map(self._predict_func, 
                  AllExtents)
            p.close()
            p.join()

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
    
    
    
    

    
