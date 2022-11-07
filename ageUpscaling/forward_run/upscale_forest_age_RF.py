#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:13:19 2019

#gdalwarp -ts 2160 4320 -dstnodata -9999 -r average -of netCDF /minerva/BGI/scratch/sbesnard/age_upscale/age_product/age_MPI_product_TC020.nc /minerva/BGI/scratch/sbesnard/age_upscale/age_product/age_MPI_product_agg_TC020.nc

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
from utils import RFregression, RFclassifier 
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
    agb                              =  xr.open_dataset("/Net/Groups/BGI/work_3/biomass/GlobBiomass/data/global/upscaling/Data_Pool/static/RSS_agb_mar_2019/agb_001deg_cc_min_010.bilinear.nc")["agb_001deg_cc_min_010"].sel(lat=list(extents.values())[0],lon=list(extents.values())[1])
    agb                              =  agb.rename({'lon': 'longitude','lat': 'latitude'}).to_dataset(name = 'agb')
    agb['latitude']                  =  AnnualMeanTemperature['latitude'] 
    agb['longitude']                 =  AnnualMeanTemperature['longitude']
    treecover                        =  xr.open_dataset("/Net/Groups/BGI/work_3/biomass/GlobBiomass/data/global/upscaling/treecover2010/treecover.median.43200.21600.nc")["treecover"].sel(lat=list(extents.values())[0],lon=list(extents.values())[1])
    treecover                        =  treecover.rename({'lon': 'longitude','lat': 'latitude'}).to_dataset(name = 'treecover')
    treecover['latitude']            =  AnnualMeanTemperature['latitude'] 
    treecover['longitude']           =  AnnualMeanTemperature['longitude']
    tree_height                      =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/canopy_height/v_2005/Data/canopy_height.43200.21600.2005.nc")["canopy_height"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1]).to_dataset(name = 'tree_height')
    tree_height['latitude']          =  AnnualMeanTemperature['latitude'] 
    tree_height['longitude']         =  AnnualMeanTemperature['longitude']
    data_cube                        =  xr.merge([agb, treecover, tree_height, AnnualMeanTemperature, AnnualPrecipitation,Isothermality,
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
    feature_reg = np.concatenate(pd.read_csv('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/feature_selection/variable_selected_borutaReg.csv').values)
    X_upscale_reg = data_cube[feature_reg].to_array().transpose('latitude', 'longitude', 'variable').values.reshape(-1,feature_reg.shape[0])

    RF_pred = np.zeros(X_upscale_class.shape[0]) * np.nan
    mask = (np.all(np.isfinite(X_upscale_class), axis=1)) & (np.all(np.isfinite(X_upscale_reg), axis=1))

    if (X_upscale_class[mask].shape[0]>0):
        #load model
        best_model_class = load('/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_RFclassifier.joblib')
        best_model_reg = load('/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_RFregression.joblib')        
        #run model
        pred_ = RFclassifier.predict_(best_model_class, X_upscale_class[mask])
        pred_[pred_==1] = 300
        pred_reg= RFregression.predict_(best_model_reg, X_upscale_reg[mask])
        pred_reg[pred_reg>=300] = 299
        pred_reg[pred_reg<0] = 0                
        pred_[pred_==0] = pred_reg[pred_==0]
        RF_pred[mask] = pred_    

    RF_pred = RF_pred.reshape(OrigShape)
    #Output the data to numpy binary files to be loaded later
    FileName = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/mean/Lat{0}x{1}_Lon{2}x{3}.npy'.format(extents['latitude'].start,extents['latitude'].stop,extents['longitude'].start,extents['longitude'].stop)
    np.save(FileName,RF_pred)
       
#%% Load forest age data    
print(f'Loading FIDC data')      
age_data = xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/training_data/training_data_ageMap_OG300_new.nc")

##classification
###############################################
#%% Create feature and target arrays
print(f'Preparing training data') 
feature_ = np.concatenate(pd.read_csv('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/feature_selection/variable_selected_borutaClass.csv').values)
X = age_data[feature_].to_array().transpose('plot', 'sample', 'variable').values
Y = age_data['age'].values
Y_class = Y.copy()
Y_class[Y==300] = 1
Y_class[Y<300] = 0

#%% Split train and valid set
n_features = X.shape[2]
X, Y_class = X.reshape(-1, n_features), Y_class.reshape(-1)
mask_train = (np.all(np.isfinite(X), axis=1)) & (np.isfinite(Y_class))
X, Y_class = X[mask_train, :], Y_class[mask_train]

# Fits the model with X and Y
#print('Model training classification')
best_model_class = RFclassifier.tune_(X, Y_class)
dump(best_model_class,'/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_RFclassifier.joblib')

## Regression
###############################################
#%% Create feature and target arrays
feature_ = np.concatenate(pd.read_csv('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/feature_selection/variable_selected_borutaReg.csv').values)
X = age_data[feature_].to_array().transpose('plot', 'sample', 'variable').values
Y = age_data['age'].where(age_data['age']<300).values
Y[Y<1] = 1

#%% Split train and valid set
n_features = X.shape[2]
X, Y = X.reshape(-1, n_features), Y.reshape(-1)
mask_train = (np.all(np.isfinite(X), axis=1)) & (np.isfinite(Y))
X, Y = X[mask_train, :], Y[mask_train]
    
# Fits the model with X and Y
#print('Model training regression')
best_model_reg = RFregression.tune_(X, Y)
dump(best_model_reg,'/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_RFregression.joblib')

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
njobs = 50
p=mp.Pool(njobs,maxtasksperchild=1)
p.map(Forward,AllExtents)
p.close()
p.join()

#%% Load numpy arrays and stack them
print('Combining chuncks and creating final product')   
RF_pred_dir = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/mean/'
fileList= os.listdir(RF_pred_dir)
for extents in range(len(AllExtents)):
    fileList[extents] = np.load('/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/mean/Lat{0}x{1}_Lon{2}x{3}.npy'.format(AllExtents[extents]['latitude'].start,AllExtents[extents]['latitude'].stop,AllExtents[extents]['longitude'].start,AllExtents[extents]['longitude'].stop)).reshape(-1)
age_productRF = np.concatenate(fileList)
age_productRF = np.array(age_productRF.reshape(21600, 43200))

# Create xarray and export ndcf file
MeanDiurnalRange = xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v1_4rel3/Data/MeanDiurnalRange.nc")
age_product = xr.Dataset(data_vars={'age':(('latitude', 'longitude'), age_productRF)},
        coords={'latitude': MeanDiurnalRange.coords["latitude"],
                'longitude': MeanDiurnalRange.coords["longitude"]})
age_product.to_netcdf('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/nc_file/age_globbiomass_TC_010_RF_withTC.nc', encoding={'age': {'dtype': np.float32,
                                                                                                                             'zlib': True, 'complevel': 9}})
#age_product.chunk({'latitude': 100, 'longitude':100}).to_zarr('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/zarr/age_globbiomass_TC_010_RF', mode='w')


