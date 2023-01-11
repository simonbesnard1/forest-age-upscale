#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:57 2022

@author: simon
"""
from ageUpscaling.diagnostic.report import Report

#%% Generate report
report_ = Report(study_dir= '/home/simon/Documents/science/GFZ/projects/forest-age-upscale/output/subsetFIA/XGBoost/version-1.7')
report_.generate_diagnostic(diagnostic_type =  {'cross-validation'})
