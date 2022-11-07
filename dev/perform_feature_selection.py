#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:54:48 2022

@author: simon
"""
import xarray as xr
from ageUpscaling.methods.feature_selection import FeatureSelection

#%% Load data
age_data = xr.open_dataset("/home/simon/Documents/science/GFZ/projects/forest_age_upscale/data/training_data/training_data_ageMap_OG300_new.nc")

#%% Create target and input data
feature_ = ['agb', "AnnualMeanTemperature", "AnnualPrecipitation", "Isothermality", "MaxTemperatureofWarmestMonth", 
            "MeanDiurnalRange", "MeanTemperatureofColdestQuarter", "MeanTemperatureofDriestQuarter",  
            "MeanTemperatureofWarmestQuarter", "MeanTemperatureofWettestQuarter", "MinTemperatureofColdestMonth", 
            "PrecipitationofColdestQuarter", "PrecipitationofDriestMonth", "PrecipitationofDriestQuarter", 
            "PrecipitationofWarmestQuarter", "PrecipitationofWettestMonth", "PrecipitationofWettestQuarter", 
            "PrecipitationSeasonality", "TemperatureAnnualRange", "TemperatureSeasonality", "srad", "vapr", "wind"]

#%% Perform feature selection
feature_method = FeatureSelection(model='regression', selection_method = "boruta")
feature_selected = feature_method.get_features(data = age_data,features=feature_, target="age")


