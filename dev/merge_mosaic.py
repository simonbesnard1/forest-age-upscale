#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:26:08 2023

@author: simon
"""

import os
import glob

files_to_mosaic = glob.glob('/home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/*_2020_AGB.tif')

files_string = " ".join(files_to_mosaic)

command = "gdal_merge.py -o /home/simon/gfz_hpc/projects/forest-age-upscale/data/global_product/20m_annual/ESA_Forest_Carbon_Monitoring/Above_ground_biomass/TEAK_Aspect_Mosaic.tif -of gtiff " + files_string
print(os.popen(command).read())
