#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:13:19 2019

#gdalwarp -ts 2160 4320 -dstnodata -9999 -r average -of netCDF /home/Simon/minerva/BGI/scratch/sbesnard/age_upscale/age_product/age_MPI_product_TC020.nc /home/Simon/minerva/BGI/scratch/sbesnard/age_upscale/age_product/age_MPI_product_agg_TC020.nc

@author: sbesnard
"""
import numpy as np
import multiprocessing as mp
import xarray as xr
from joblib import dump, load
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/Net/Groups/BGI/work_2/FIDC_age_upscale/code/upscaling_model')
from utils import RFclassifier 
import pandas as pd

def Forward(extents={'latitude':slice(51,50),'longitude':slice(30,31)}):
    
    #Load gridded product
    AnnualMeanTemperature            =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/AnnualMeanTemperature.nc")["AnnualMeanTemperature"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    AnnualPrecipitation              =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/AnnualPrecipitation.nc")["AnnualPrecipitation"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    Isothermality                    =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/Isothermality.nc")["Isothermality"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MaxTemperatureofWarmestMonth     =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MaxTemperatureofWarmestMonth.nc")["MaxTemperatureofWarmestMonth"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MeanDiurnalRange                 =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MeanDiurnalRange.nc")["MeanDiurnalRange"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MeanTemperatureofColdestQuarter  =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MeanTemperatureofColdestQuarter.nc")["MeanTemperatureofColdestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MeanTemperatureofDriestQuarter   =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MeanTemperatureofDriestQuarter.nc")["MeanTemperatureofDriestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MeanTemperatureofWarmestQuarter  =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MeanTemperatureofWarmestQuarter.nc")["MeanTemperatureofWarmestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MeanTemperatureofWettestQuarter  =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MeanTemperatureofWettestQuarter.nc")["MeanTemperatureofWettestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MinTemperatureofColdestMonth     =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MinTemperatureofColdestMonth.nc")["MinTemperatureofColdestMonth"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofColdestQuarter    =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofColdestQuarter.nc")["PrecipitationofColdestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofDriestMonth       =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofDriestMonth.nc")["PrecipitationofDriestMonth"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofDriestQuarter     =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofDriestQuarter.nc")["PrecipitationofDriestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofWarmestQuarter    =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofWarmestQuarter.nc")["PrecipitationofWarmestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofWettestMonth      =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofWettestMonth.nc")["PrecipitationofWettestMonth"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofWettestQuarter    =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofWettestQuarter.nc")["PrecipitationofWettestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationSeasonality         =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationSeasonality.nc")["PrecipitationSeasonality"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    TemperatureAnnualRange           =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/TemperatureAnnualRange.nc")["TemperatureAnnualRange"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    TemperatureSeasonality           =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/TemperatureSeasonality.nc")["TemperatureSeasonality"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1]) / 100
    srad                             =  xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/global_product/srad_mean_worldClim.nc")["srad"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1]) * 0.001 * 11.574
    srad['latitude']                 =  AnnualMeanTemperature['latitude'] 
    srad['longitude']                =  AnnualMeanTemperature['longitude']
    wind                             =  xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/global_product/wind_mean_worldClim.nc")["wind"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    wind['latitude']                 =  AnnualMeanTemperature['latitude'] 
    wind['longitude']                =  AnnualMeanTemperature['longitude']    
    vapr                             =  xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/global_product/vapr_mean_worldClim.nc")["vapr"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1]) * 10
    vapr['latitude']                 =  AnnualMeanTemperature['latitude'] 
    vapr['longitude']                =  AnnualMeanTemperature['longitude']
    agb                              =  xr.open_dataset("/Net/Groups/BGI/work_3/biomass/GlobBiomass/data/global/upscaling/Data_Pool/static/RSS_agb_mar_2019/agb_001deg_cc_min_020.bilinear.nc")["agb_001deg_cc_min_020"].sel(lat=list(extents.values())[0],lon=list(extents.values())[1])
    agb                              =  agb.rename({'lon': 'longitude','lat': 'latitude'}).to_dataset(name = 'agb')
    agb['latitude']                  =  AnnualMeanTemperature['latitude'] 
    agb['longitude']                 =  AnnualMeanTemperature['longitude']
    data_cube                        =  xr.merge([agb, AnnualMeanTemperature, AnnualPrecipitation,Isothermality,
                                                MaxTemperatureofWarmestMonth, MeanDiurnalRange, MeanTemperatureofColdestQuarter,
                                                MeanTemperatureofDriestQuarter, MeanTemperatureofWarmestQuarter, MeanTemperatureofWettestQuarter,
                                                MinTemperatureofColdestMonth, PrecipitationofColdestQuarter, PrecipitationofDriestMonth,
                                                PrecipitationofDriestQuarter, PrecipitationofWarmestQuarter, PrecipitationofWettestMonth,
                                                PrecipitationofWettestQuarter, PrecipitationSeasonality, PrecipitationSeasonality,
                                                TemperatureAnnualRange, TemperatureSeasonality, srad, wind, vapr])
 
    # Save the original shape of the data
    OrigShape = MeanTemperatureofWettestQuarter.shape
    feature_class = np.concatenate(pd.read_csv('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/feature_selection/variable_selected_borutaClass.csv').values)
    #X_upscale_class = np.stack((agb.values.reshape(-1),
    #                            AnnualMeanTemperature.values.reshape(-1),
    #                            AnnualPrecipitation.values.reshape(-1), 
    #                            Isothermality.values.reshape(-1),
    #                            AnnualSrad.values.reshape(-1),
    #                            AnnualVapr.values.reshape(-1),
    #                            AnnualWind.values.reshape(-1))).T
    X_upscale_class = data_cube[feature_class].to_array().transpose('latitude', 'longitude', 'variable').values.reshape(-1,feature_class.shape[0])

    oldgrowth_pred, nonoldgrowth_pred = np.zeros(X_upscale_class.shape[0]) * np.nan, np.zeros(X_upscale_class.shape[0]) * np.nan
    mask = np.all(np.isfinite(X_upscale_class), axis=1)
    if (X_upscale_class[mask].shape[0]>0):
        #load model
        best_model_class = load('/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_RFclassifier.joblib')
        #run model
        pred_ = best_model_class.predict(X_upscale_class[mask])
        proba_oldgrowth = best_model_class.predict_proba(X_upscale_class[mask])[:,1]
        proba_oldgrowth[pred_== 0] = np.nan
        proba_nonoldgrowth = best_model_class.predict_proba(X_upscale_class[mask])[:,0]
        proba_nonoldgrowth[pred_== 1] = np.nan        
        oldgrowth_pred[mask] = proba_oldgrowth    
        nonoldgrowth_pred[mask] = proba_nonoldgrowth
    nonoldgrowth_pred = nonoldgrowth_pred.reshape(OrigShape)
    oldgrowth_pred = oldgrowth_pred.reshape(OrigShape)    
    #Output the data to numpy binary files to be loaded later
    FileName = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/oldgrowth/Lat{0}x{1}_Lon{2}x{3}.npy'.format(extents['latitude'].start,extents['latitude'].stop,extents['longitude'].start,extents['longitude'].stop)
    np.save(FileName,oldgrowth_pred)
    FileName = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/nonoldgrowth/Lat{0}x{1}_Lon{2}x{3}.npy'.format(extents['latitude'].start,extents['latitude'].stop,extents['longitude'].start,extents['longitude'].stop)
    np.save(FileName,nonoldgrowth_pred)
#%% Run upscaling
print('Upscaling procedure')   
nLatChunks = 50
nLonChunks = 2
LatChunks  = np.linspace(90,-90,nLatChunks)
LonChunks  = np.linspace(-180,180,nLonChunks)
AllExtents = []
for lat in range(nLatChunks-1):
    for lon in range(nLonChunks-1):
        AllExtents.append({'latitude':slice(LatChunks[lat],LatChunks[lat+1]),'longitude':slice(LonChunks[lon],LonChunks[lon+1])})
njobs = 16
p=mp.Pool(njobs,maxtasksperchild=1)
p.map(Forward,AllExtents)
p.close()
p.join()

#%% Load numpy arrays and stack them
print('Combining chuncks and creating final product')   
RF_pred_dir = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/oldgrowth/'
fileList= os.listdir(RF_pred_dir)
for extents in range(len(AllExtents)):
    fileList[extents] = np.load('/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/oldgrowth/Lat{0}x{1}_Lon{2}x{3}.npy'.format(AllExtents[extents]['latitude'].start,AllExtents[extents]['latitude'].stop,AllExtents[extents]['longitude'].start,AllExtents[extents]['longitude'].stop)).reshape(-1)
proba_oldgrowth = np.concatenate(fileList)
proba_oldgrowth = np.array(proba_oldgrowth.reshape(21600, 43200))

RF_pred_dir = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/nonoldgrowth/'
fileList= os.listdir(RF_pred_dir)
for extents in range(len(AllExtents)):
    fileList[extents] = np.load('/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/nonoldgrowth/Lat{0}x{1}_Lon{2}x{3}.npy'.format(AllExtents[extents]['latitude'].start,AllExtents[extents]['latitude'].stop,AllExtents[extents]['longitude'].start,AllExtents[extents]['longitude'].stop)).reshape(-1)
proba_nonoldgrowth = np.concatenate(fileList)
proba_nonoldgrowth = np.array(proba_nonoldgrowth.reshape(21600, 43200))

# Create xarray and export ndcf file
MeanDiurnalRange = xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v1_4rel3/Data/MeanDiurnalRange.nc")
proba_oldgrowth = xr.Dataset(data_vars={'proba_oldgrowth':(('latitude', 'longitude'), proba_oldgrowth)},
        coords={'latitude': MeanDiurnalRange.coords["latitude"],
                'longitude': MeanDiurnalRange.coords["longitude"]})
proba_oldgrowth.to_netcdf('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/classProbaOG_globbiomass_TC_010_RF.nc', encoding={'proba_oldgrowth': {'dtype': np.float32,
                                                                                                                             'zlib': True, 'complevel': 9}})
proba_nonoldgrowth = xr.Dataset(data_vars={'proba_nonoldgrowth':(('latitude', 'longitude'), proba_nonoldgrowth)},
        coords={'latitude': MeanDiurnalRange.coords["latitude"],
                'longitude': MeanDiurnalRange.coords["longitude"]})
proba_nonoldgrowth.to_netcdf('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/classProbaNoOG_globbiomass_TC_010_RF.nc',encoding={'proba_nonoldgrowth': {'dtype': np.float32,
                                                                                                                             'zlib': True, 'complevel': 9}})
