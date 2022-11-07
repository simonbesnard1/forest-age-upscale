import os
from typing import Any
import numpy as np
import pickle
import optuna
from sklearn.metrics import mean_squared_error
from ageUpscaling.dataloaders.ml_dataloader import MLDataModule

class MLMethod:
    """MLMethod class to be subclassed for method definition.

    Subclass must define `__init__` method that defines a MLModel and assigns it to the atribute `model`.
    `Subclass.__init__` must call `super().__init__(save_dir=save_dir)`.

    """
    
    def __init__(
            self,
            save_dir: str) -> None:

        self.save_dir = save_dir
        self._model = None
        
    def get_datamodule(
            self,
            cube_path:str, 
            data_config:dict,
            train_subset: dict[str, Any] = {},
            valid_subset: dict[str, Any] = {},
            test_subset: dict[str, Any] = {},
            **kwargs) -> dict:
        
        mlData = MLDataModule(cube_path, 
                              data_config,
                              train_subset, 
                              valid_subset, 
                              test_subset)

        return mlData
        
    def tune_(self, 
              cube_path:np.array = [], 
              data_config:np.array = [], 
              train_subset:dict={},
              valid_subset:dict={},
              iter_: int=1, 
              hyper_params:dict= {}) -> None:

        mldata = self.get_datamodule(cube_path=cube_path,
                                     data_config=data_config,
                                     train_subset=train_subset,
                                     valid_subset=valid_subset)

        if not os.path.exists(self.save_dir + '/save_model/MLPRegressor/fold' + str(iter_)):
            os.makedirs(self.save_dir + '/save_model/MLPRegressor/fold' + str(iter_))
        
        study = optuna.create_study(study_name = 'XvalAge_fold' + str(iter_), 
                                    storage='sqlite:///' + self.save_dir + '/save_model/MLPRegressor/fold' + str(iter_) + '/hp_trial.db',
                                    pruner= optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', 
                                                                                   reduction_factor=4, 
                                                                                   min_early_stopping_rate=0),
                                    direction='minimize')
        study.optimize(lambda trial: self._objective(trial, mldata, hyper_params, iter_, self.save_dir), 
                       n_trials=300, n_jobs=4)
        
        with open(self.save_dir + '/save_model/model_trial' + "{}.pickle".format(study.best_trial.number), "rb") as fin:
            self.best_model = pickle.load(fin)
            
            
    def _objective(self, 
                   trial: optuna.Trial,
                   hyper_params:dict,
                   mldata:dict,
                   save_dir:str) -> float:
        
        self.model(hyper_params, save_dir).fit(mldata.train_dataloader().get_xy()['features'], mldata.train_dataloader().get_xy()['target'])
        
        with open(save_dir + '/save_model/model_trial{}.pickle'.format(trial.number), "wb") as fout:
            pickle.dump(self.model, fout)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return mean_squared_error(mldata.valid_dataloader()['target'], self.model.predict(mldata.valid_dataloader()['features']), squared=False)
    
    def predict(self,
                mldata:dict) -> np.array:
            
        pred_ = self.best_model.predict(mldata.test_dataloader().get_xy()['features'])
                
        return pred_

    