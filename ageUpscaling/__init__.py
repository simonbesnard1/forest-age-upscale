# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Simon Besnard
# SPDX-License-Identifier: EUPL-1.2 
# Version :   1.0
# Contact :   besnard@gfz-potsdam.de

__title__ = 'ageUpscaling'
__version__ = '1.0'
__author__ = 'Simon Besnard'
__author_email__ = 'besnard@gfz-potsdam.de'
__license__ = 'EUPL-1.2'
__copyright__ = '2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences'

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



