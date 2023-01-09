__title__ = 'ageUpscaling'
__version__ = '0.0.0'
__author__ = 'Simon Besnard'
__author_email__ = 'besnard.sim@gmail.com'
__license__ = 'GNU GPLv3'
__copyright__ = 'Copyright 2022 by GFZ-Potsdam'

from . import methods, utils, core, dataloaders, fidc_cube, upscaling, transformers, diagnostic

__all__ = [
    'methods',
    "utils",
    "core",
    "dataloaders", 
    "fidc_cube", 
    "global_cube",
    "upscaling",
    "transformers",
    "diagnostic"   
]
