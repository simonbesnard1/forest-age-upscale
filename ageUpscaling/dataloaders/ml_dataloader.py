from typing import Any
import numpy as np
import xarray as xr
from ageUpscaling.dataloaders.base import MLData

class MLDataModule:
    """Define dataloaders.

    
    Parameters:
        cube_path: str
            Path to the datacube.
        DataConfig: DataConfig
            The data configuration.
        train_subset: Dict[str, Any]:
            Training set selection.
        valid_subset: Dict[str, Any]:
            Same as `train_subset`, for validation set.
        test_subset: Dict[str, Any]:
            Same as `train_subset`, for test set.
        **kwargs:
            Keyword arguments passed to `DataLoader`, same for training, validation and test set loader.
    """
    def __init__(
            self,
            DataConfig: dict[str, Any] = {},
            target: dict[str, Any] = {},
            features: dict[str, Any] = {},            
            train_subset: dict[str, Any] = {},
            valid_subset: dict[str, Any] = {},
            test_subset: dict[str, Any] = {},
            norm_stats: dict[str, dict[str, float]] = {},
            **kwargs) -> None:
        super().__init__()

        self.DataConfig = DataConfig
        self.target = target
        self.features = features        
        self.train_subset = train_subset
        self.valid_subset = valid_subset
        self.test_subset = test_subset
        self.norm_stats = norm_stats
        self._kwargs = kwargs
        
        if len(self.norm_stats) == 0:

            for var in  self.target + self.features:
                data = xr.open_dataset(self.DataConfig['training_dataset']).sel(cluster = train_subset)[var]
                data_mean = data.mean().compute().item()
                data_std = data.std().compute().item()
                self.norm_stats[var] = {'mean': data_mean, 'std': data_std}

    def train_dataloader(self) -> np.array:
        """Returns the training dataloader."""


        train_data = MLData(self.DataConfig, self.target, self.features, self.train_subset, self.norm_stats)        
            
        return train_data

    def val_dataloader(self) -> np.array:
        """Returns the validation dataloader."""

        valid_data = MLData(self.DataConfig, self.target, self.features, self.valid_subset, self.norm_stats)
            
        return valid_data  

    def test_dataloader(self) -> np.array:
        """Returns the test dataloader."""

        test_data = MLData(self.DataConfig, self.target, self.features, self.test_subset, self.norm_stats)
            
        return test_data

    
