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


DEFAULT_VARS = {"age",
                "agb",
                "AnnualMeanTemperature_WorlClim" ,
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
                "AnnualVapr_WorlClim"}

DEFAULT_LONG_NAMES = {
        "age"  : "forest age at plot level",
        "agb"  : "above-ground biomass",
        "AnnualMeanTemperature_WorlClim" : "Annual Mean Temperature - worldclim dataset",
        "MeanDiurnalRange_WorlClim" : " Mean Diurnal Range (Mean of monthly (max temp - min temp)) - worldclim dataset",
        "TemperatureSeasonality_WorlClim" : "Temperature Seasonality (standard deviation *100) - worldclim dataset",
        "MaxTemperatureofWarmestMonth_WorlClim" : "Max Temperature of Warmest Month - worldclim dataset",
        "MinTemperatureofColdestMonth_WorlClim" : "Min Temperature of Coldest Month - worldclim dataset",
        "TemperatureAnnualRange_WorlClim" :  "Temperature Annual Range (MaxTemperatureofWarmestMonth - MinTemperatureofColdestMonth) - worldclim dataset",
        "MeanTemperatureofWettestQuarter_WorlClim" : "Mean Temperature of Wettest Quarter - worldclim dataset",
        "MeanTemperatureofDriestQuarter_WorlClim"  : "Mean Temperature of Driest Quarter - worldclim dataset",
        "MeanTemperatureofWarmestQuarter_WorlClim" : "Mean Temperature of Warmest Quarter - worldclim dataset",
        "MeanTemperatureofColdestQuarter_WorlClim" : " Mean Temperature of Coldest Quarter - worldclim dataset",        
        "Isothermality_WorlClim" : "Isothermality (MeanDiurnalRange - TemperatureAnnualRange) *100 - worldclim dataset",
        "AnnualPrecipitation_WorlClim" : "Annual Precipitation - worldclim dataset",
        "PrecipitationofWettestMonth_WorlClim"  : "Precipitation of Wettest Month - worldclim dataset",
        "PrecipitationofDriestMonth_WorlClim" :  "Precipitation of Driest Month - worldclim dataset",
        "PrecipitationSeasonality_WorlClim" : "Precipitation Seasonality (Coefficient of Variation) - worldclim dataset",
        "PrecipitationofWettestQuarter_WorlClim" : "Precipitation of Wettest Quarter - worldclim dataset",
        "PrecipitationofDriestQuarter_WorlClim" : "Precipitation of Driest Quarter - worldclim dataset",
        "PrecipitationofWarmestQuarter_WorlClim" : "Precipitation of Warmest Quarter - worldclim dataset", 
        "PrecipitationofColdestQuarter_WorlClim" : "Precipitation of Coldest Quarter - worldclim dataset",
        "AnnualSrad_WorlClim" : "Annual Mean solar radiation - worldclim dataset",
        "AnnualWind_WorlClim" : "Annual Mean wind speed - worldclim dataset",
        "AnnualVapr_WorlClim" : "Annual Mean water vapor pressure - worldclim dataset"}

DEFAULT_UNITS = {"age"  : "years",
        "agb"  : "Mg ha-1",
        "AnnualMeanTemperature_WorlClim" : "deg C",
        "MeanDiurnalRange_WorlClim" : "deg C",
        "TemperatureSeasonality_WorlClim" : "deg C",
        "MaxTemperatureofWarmestMonth_WorlClim" : "deg C",
        "MinTemperatureofColdestMonth_WorlClim" : "deg C",
        "TemperatureAnnualRange_WorlClim" :  "deg C",
        "MeanTemperatureofWettestQuarter_WorlClim" : "deg C",
        "MeanTemperatureofDriestQuarter_WorlClim"  : "deg C",
        "MeanTemperatureofWarmestQuarter_WorlClim" : "deg C",
        "MeanTemperatureofColdestQuarter_WorlClim" : "deg C",        
        "Isothermality_WorlClim" : "deg C",
        "AnnualPrecipitation_WorlClim" : "mm",
        "PrecipitationofWettestMonth_WorlClim"  : "mm",
        "PrecipitationofDriestMonth_WorlClim" :  "mm",
        "PrecipitationSeasonality_WorlClim" : "mm",
        "PrecipitationofWettestQuarter_WorlClim" : "mm",
        "PrecipitationofDriestQuarter_WorlClim" : "mm",
        "PrecipitationofWarmestQuarter_WorlClim" : "mm", 
        "PrecipitationofColdestQuarter_WorlClim" : "mm",
        "AnnualSrad_WorlClim" : "W m-2",
        "AnnualWind_WorlClim" : "m s-1",        
        "AnnualVapr_WorlClim" : "hPa"}

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

    def compute_cube(self, 
                     variables:dict= 'default'):
        if variables == 'default':
            var_names = DEFAULT_VARS
        
        df_ = pd.read_csv(self.input_csv)
        sites = df_.cluster.values
        plot_ds = []
        for site in np.unique(sites):
            siteMask  = site==sites
            coords = {'cluster': [site], 'sample':np.arange(len(df_['agb'].values[siteMask]))}
            ds = {}
            for _var in var_names:
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
