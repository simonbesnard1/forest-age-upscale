#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:49:25 2019

@author: simon
"""
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import os
CurrentScript = os.path.basename('/Net/Groups/BGI/work_2/FIDC_age_upscale/code/data/create_final_dataset.py')
out_dir = '/home/simon/Documents/science/GFZ/projects/forest_age_upscale/data/training_data/'

# load dataset
df_ = pd.read_csv("/home/simon/Documents/science/GFZ/projects/forest_age_upscale/data/training_data/training_data_ageMap_OG300.csv")
df_ = df_.dropna()

# Define long names
long_names = {
        "age"  : "forest age at plot level",
        "agb"  : "above-ground biomass",
        #"tree_height" : "tree height",
        "AnnualMeanTemperature" : "Annual Mean Temperature - worldclim dataset",
        "MeanDiurnalRange" : " Mean Diurnal Range (Mean of monthly (max temp - min temp)) - worldclim dataset",
        "TemperatureSeasonality" : "Temperature Seasonality (standard deviation *100) - worldclim dataset",
        "MaxTemperatureofWarmestMonth" : "Max Temperature of Warmest Month - worldclim dataset",
        "MinTemperatureofColdestMonth" : "Min Temperature of Coldest Month - worldclim dataset",
        "TemperatureAnnualRange" :  "Temperature Annual Range (MaxTemperatureofWarmestMonth - MinTemperatureofColdestMonth) - worldclim dataset",
        "MeanTemperatureofWettestQuarter" : "Mean Temperature of Wettest Quarter - worldclim dataset",
        "MeanTemperatureofDriestQuarter"  : "Mean Temperature of Driest Quarter - worldclim dataset",
        "MeanTemperatureofWarmestQuarter" : "Mean Temperature of Warmest Quarter - worldclim dataset",
        "MeanTemperatureofColdestQuarter" : " Mean Temperature of Coldest Quarter - worldclim dataset",        
        "Isothermality" : "Isothermality (MeanDiurnalRange - TemperatureAnnualRange) *100 - worldclim dataset",
        "AnnualPrecipitation" : "Annual Precipitation - worldclim dataset",
        "PrecipitationofWettestMonth"  : "Precipitation of Wettest Month - worldclim dataset",
        "PrecipitationofDriestMonth" :  "Precipitation of Driest Month - worldclim dataset",
        "PrecipitationSeasonality" : "Precipitation Seasonality (Coefficient of Variation) - worldclim dataset",
        "PrecipitationofWettestQuarter" : "Precipitation of Wettest Quarter - worldclim dataset",
        "PrecipitationofDriestQuarter" : "Precipitation of Driest Quarter - worldclim dataset",
        "PrecipitationofWarmestQuarter" : "Precipitation of Warmest Quarter - worldclim dataset", 
        "PrecipitationofColdestQuarter" : "Precipitation of Coldest Quarter - worldclim dataset",
        "AnnualSrad" : "Annual Mean solar radiation - worldclim dataset",
        "AnnualWind" : "Annual Mean wind speed - worldclim dataset",
        "AnnualVapr" : "Annual Mean water vapor pressure - worldclim dataset"}

# Define units
units = {"age"  : "years",
        "agb"  : "Mg ha-1",
        #"tree_height" : "meter",
        "AnnualMeanTemperature" : "deg C",
        "MeanDiurnalRange" : "deg C",
        "TemperatureSeasonality" : "deg C",
        "MaxTemperatureofWarmestMonth" : "deg C",
        "MinTemperatureofColdestMonth" : "deg C",
        "TemperatureAnnualRange" :  "deg C",
        "MeanTemperatureofWettestQuarter" : "deg C",
        "MeanTemperatureofDriestQuarter"  : "deg C",
        "MeanTemperatureofWarmestQuarter" : "deg C",
        "MeanTemperatureofColdestQuarter" : "deg C",        
        "Isothermality" : "deg C",
        "AnnualPrecipitation" : "mm",
        "PrecipitationofWettestMonth"  : "mm",
        "PrecipitationofDriestMonth" :  "mm",
        "PrecipitationSeasonality" : "mm",
        "PrecipitationofWettestQuarter" : "mm",
        "PrecipitationofDriestQuarter" : "mm",
        "PrecipitationofWarmestQuarter" : "mm", 
        "PrecipitationofColdestQuarter" : "mm",
        "AnnualSrad" : "W m-2",
        "AnnualWind" : "m s-1",        
        "AnnualVapr" : "hPa"}


sites = df_.cluster.values

# build list of sub-arrays
plot_ds = []
for site in np.unique(sites):
    siteMask  = site==sites
    coords = {'cluster': [site], 'sample':np.arange(len(df_['agb'].values[siteMask]))}
    ds = {}
    for _var in long_names.keys():
        ds[_var] = (('cluster', 'sample'), [df_[_var].values[siteMask]])
    ds = xr.Dataset(data_vars=ds, coords=coords)  
    #ds = ds.expand_dims({'cluster':[site]})  
    ds = ds.assign_coords(latitude  =  np.unique(df_['latitude_origin'].values[siteMask]),
                          longitude = np.unique(df_['longitude_origin'].values[siteMask]))
    plot_ds.append(ds)    
plot_ds = xr.concat(plot_ds, dim= 'cluster')
for _var in long_names.keys():
    plot_ds[_var] = plot_ds[_var].assign_attrs(long_name=long_names[_var],
                                               units=units[_var])
plot_ds = plot_ds.assign_attrs(title = "Training dataset for stand age upscaling",
                     created_by='Simon Besnard',
                     contact = 'besnard@gfz-potsdam.de',
                     creation_date=datetime.now().strftime("%d-%m-%Y %H:%M"))
plot_ds.to_netcdf(out_dir + '/training_data_ageMap_OG300.nc', mode='w')
