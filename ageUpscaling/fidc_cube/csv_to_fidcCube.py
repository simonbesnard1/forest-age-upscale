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
from abc import ABC
from typing import Any

DEFAULT_LONG_NAMES = {
        "age"  : "forest age at plot level",
        "agb"  : "above-ground biomass",
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

DEFAULT_UNITS = {"age"  : "years",
        "agb"  : "Mg ha-1",
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

class ImportAndSave(ABC):
    """ImportAndSave
    
    Imports FIDC formatted csv file.
    
    """
    def __init__(
        self,
        input_csv: str = '',
        out_file: str = None,
        variables: dict[str, Any] = DEFAULT_LONG_NAMES,
        units: dict[str, Any] = DEFAULT_UNITS):
    
        super().__init__()
        
        self.input_csv = input_csv
        self.out_file = out_file
        self.variables = variables
        self.units = units

    def run(self):
        
        df_ = pd.read_csv(self.input_csv)
        df_ = df_.dropna()
        sites = df_.cluster.values
        
        plot_ds = []
        for site in np.unique(sites):
            siteMask  = site==sites
            coords = {'cluster': [site], 'sample':np.arange(len(df_['agb'].values[siteMask]))}
            ds = {}
            for _var in DEFAULT_LONG_NAMES.keys():
                ds[_var] = (('cluster', 'sample'), [df_[_var].values[siteMask]])
            ds = xr.Dataset(data_vars=ds, coords=coords)  
            ds = ds.assign_coords(latitude  =  np.unique(df_['latitude_origin'].values[siteMask]),
                                  longitude = np.unique(df_['longitude_origin'].values[siteMask]))
            plot_ds.append(ds)    
        plot_ds = xr.concat(plot_ds, dim= 'cluster')
        
        for _var in DEFAULT_LONG_NAMES.keys():
            plot_ds[_var] = plot_ds[_var].assign_attrs(long_name=DEFAULT_LONG_NAMES[_var],
                                                       units=DEFAULT_UNITS[_var])
        plot_ds = plot_ds.assign_attrs(title = "Training dataset for stand age upscaling",
                             created_by='Simon Besnard',
                             contact = 'besnard@gfz-potsdam.de',
                             creation_date=datetime.now().strftime("%d-%m-%Y %H:%M"))
        plot_ds.to_netcdf(self.out_file, mode='w')
        
        return plot_ds
