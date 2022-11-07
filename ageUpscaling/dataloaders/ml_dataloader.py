from typing import Any
import numpy as np
from ageUpscaling.dataloaders.base import MLData


class MLDataModule:
    """Define dataloaders.

    
    Parameters:
        cube_path: str
            Path to the datacube.
        data_config: DataConfig
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
            cube_path: str,
            data_config: dict[str, Any] = {},
            train_subset: dict[str, Any] = {},
            valid_subset: dict[str, Any] = {},
            test_subset: dict[str, Any] = {},
            **kwargs) -> None:
        super().__init__()

        
        self.cube_path = cube_path
        self.data_config = data_config
        self.train_subset = train_subset
        self.valid_subset = valid_subset
        self.test_subset = test_subset
        self._kwargs = kwargs

    def train_dataloader(self) -> np.array:
        """Returns the training dataloader."""


        train_data = MLData(self.cube_path,self.train_subset, self.data_config)
            
        return train_data

    def val_dataloader(self) -> np.array:
        """Returns the validation dataloader."""

        valid_data = MLData(self.cube_path,self.valid_subset, self.data_config)
            
        return valid_data  

    def test_dataloader(self) -> np.array:
        """Returns the test dataloader."""

        test_data = MLData(self.cube_path,self.test_subset, self.data_config)
            
        return test_data

    
