#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:58:01 2023

@author: simon
"""



from sklearn.cluster import BisectingKMeans
import numpy as np
import pandas as pd


ds_ = pd.read_csv("/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_v3_WithworlClim.csv")
X = ds_[["AnnualMeanTemperature_WorlClim" ,
"MeanDiurnalRange_WorlClim",
"TemperatureSeasonality_WorlClim",
"MaxTemperatureofWarmestMonth_WorlClim",
"MinTemperatureofColdestMonth_WorlClim",
"TemperatureAnnualRange_WorlClim",
"MeanTemperatureofWettestQuarter_WorlClim",
"MeanTemperatureofDriestQuarter_WorlClim" ,
"MeanTemperatureofWarmestQuarter_WorlClim",
"MeanTemperatureofColdestQuarter_WorlClim",        
"Isothermality_WorlClim",
"AnnualPrecipitation_WorlClim",
"PrecipitationofWettestMonth_WorlClim",
"PrecipitationofDriestMonth_WorlClim",
"PrecipitationSeasonality_WorlClim",
"PrecipitationofWettestQuarter_WorlClim",
"PrecipitationofDriestQuarter_WorlClim",
"PrecipitationofWarmestQuarter_WorlClim", 
"PrecipitationofColdestQuarter_WorlClim",
"AnnualSrad_WorlClim",
"AnnualWind_WorlClim",
"AnnualVapr_WorlClim"]].dropna().to_numpy()

bisect_means = BisectingKMeans(n_clusters=20, random_state=0).fit(X)

bisect_means.labels_
