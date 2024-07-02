"""
core

This module provides tools for the creation, management, and analysis of data cubes,
including functionalities for cross-validation, model training, and prediction.

Submodules:
-----------
cube : Provides the DataCube class for handling regularized cube zarr files.
study : Provides the Study class for cross-validation, model training, and prediction.
cube_utils : Provides utility functions and classes for working with data cubes.

Example usage:
--------------
from ageUpscaling.core import DataCube, Study

# Create a data cube
cube_config = {...}
data_cube = DataCube(cube_config)

# Perform a study
study_config = {...}
my_study = Study(**study_config)
"""

from ageUpscaling.core.cube import DataCube
from ageUpscaling.core.study import Study
from ageUpscaling.core.cube_utils import ComputeCube

__all__ = [
    'DataCube',
    'Study',
    'ComputeCube'
]
