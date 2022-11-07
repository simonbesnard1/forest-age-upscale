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
from utils import hp_tune_reg, hp_tune_class 

def Forward(extents={'latitude':slice(51,50),'longitude':slice(30,31)}):
    #Load gridded product
    agb                                 =  xr.open_dataset("/Net/Groups/BGI/work_3/biomass/GlobBiomass/data/global/upscaling/Data_Pool/static/RSS_agb_mar_2019/agb_001deg_cc_min_030.bilinear.nc")["agb_001deg_cc_min_030"].sel(lat=list(extents.values())[0],lon=list(extents.values())[1])
    AnnualSrad                          =  xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/global_product/srad_mean_worldClim.nc")["srad"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    TemperatureAnnualRange              =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/TemperatureAnnualRange.nc")["TemperatureAnnualRange"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    AnnualWind                          =  xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/global_product/wind_mean_worldClim.nc")["wind"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    tree_height                         =  xr.open_dataset("/Net/Groups/BGI/work_3/biomass/GlobBiomass/data/forest_height_2019/forest_height_2019.med.geo.1km_hybrid.nc")["canopy_height"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    Isothermality                       =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/Isothermality.nc")["Isothermality"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationSeasonality            =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationSeasonality.nc")["PrecipitationSeasonality"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    AnnualVapr                          =  xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/global_product/vapr_mean_worldClim.nc")["vapr"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MeanDiurnalRange                    =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MeanDiurnalRange.nc")["MeanDiurnalRange"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofDriestMonth          =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofDriestMonth.nc")["PrecipitationofDriestMonth"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MeanTemperatureofWarmestQuarter     =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MeanTemperatureofWarmestQuarter.nc")["MeanTemperatureofWarmestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    PrecipitationofWarmestQuarter       =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/PrecipitationofWarmestQuarter.nc")["PrecipitationofWarmestQuarter"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    MaxTemperatureofWarmestMonth        =  xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/MaxTemperatureofWarmestMonth.nc")["MaxTemperatureofWarmestMonth"].sel(latitude=list(extents.values())[0],longitude=list(extents.values())[1])
    
    #Save the original shape of the data
    OrigShape = MaxTemperatureofWarmestMonth.shape
    X_upscale_class = np.stack((agb.values.reshape(-1),
                                AnnualSrad.values.reshape(-1) * 0.001 * 11.574,
                                TemperatureAnnualRange.values.reshape(-1),
                                AnnualWind.values.reshape(-1),
                                tree_height.values.reshape(-1),
                                Isothermality.values.reshape(-1),
                                PrecipitationSeasonality.values.reshape(-1),
                                AnnualVapr.values.reshape(-1) * 10,
                                MeanDiurnalRange.values.reshape(-1),
                                PrecipitationofDriestMonth.values.reshape(-1))).T

    X_upscale_reg = np.stack((agb.values.reshape(-1),
                                AnnualSrad.values.reshape(-1) * 0.001 * 11.574,
                                PrecipitationSeasonality.values.reshape(-1),
                                tree_height.values.reshape(-1),
                                MeanDiurnalRange.values.reshape(-1),
                                AnnualVapr.values.reshape(-1) * 10,
                                MeanTemperatureofWarmestQuarter.values.reshape(-1),
                                PrecipitationofWarmestQuarter.values.reshape(-1),
                                MaxTemperatureofWarmestMonth.values.reshape(-1),
                                Isothermality.values.reshape(-1))).T

    RF_pred = np.zeros(X_upscale_class.shape[0]) * np.nan
    mask = np.all(np.isfinite(X_upscale_class), axis=1)
    if (X_upscale_class[mask].shape[0]>0):
        #load model
        best_model_class = load('/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_class.joblib')
        best_model_reg = load('/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_reg.joblib')        
        #run model
        pred_ = hp_tune_class.predict_(best_model_class, X_upscale_class[mask])
        pred_[pred_==1] = 300
        pred_reg= hp_tune_reg.predict_(best_model_reg, X_upscale_reg[mask])
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
age_data = xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/training_data/training_data_ageMap_OG300.nc")

##classification
###############################################
#%% Create feature and target arrays
print(f'Preparing training data')   
feature_ = ["agb", "AnnualSrad", "TemperatureAnnualRange", "AnnualWind",
           "tree_height", "Isothermality", "PrecipitationSeasonality", "AnnualVapr",
           "MeanDiurnalRange", "PrecipitationofDriestMonth"]
X = age_data[feature_].to_array().transpose('plot', 'sample_plot', 'variable').values
Y = age_data['age'].values
Y_class = Y.copy()
Y_class[Y==300] = 1
Y_class[Y<300] = 0

#%% Split train and valid set
n_features = X.shape[2]
train_indx, valid_indx = train_test_split(np.arange(Y_class.shape[0]), test_size = 0.3)
X_train, Y_train, X_valid, Y_valid = X[train_indx], Y_class[train_indx], X[valid_indx], Y_class[valid_indx]
X_train, Y_train, X_valid, Y_valid = X_train.reshape(-1, n_features), Y_train.reshape(-1), X_valid.reshape(-1, n_features), Y_valid.reshape(-1)
mask_train = (np.all(np.isfinite(X_train), axis=1)) & (np.isfinite(Y_train))
X_train, Y_train = X_train[mask_train, :], Y_train[mask_train]
mask_valid = (np.all(np.isfinite(X_valid), axis=1)) & (np.isfinite(Y_valid))
X_valid, Y_valid = X_valid[mask_valid, :], Y_valid[mask_valid]
    
# Fits the model with X and Y
print('Model training classification')
#best_model_class = hp_tune_class.tune_(X_train, Y_train, X_valid, Y_valid)
#dump(best_model_class,'/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_class.joblib')

## Regression
###############################################
#%% Create feature and target arrays
feature_ = ["agb", "AnnualSrad", "PrecipitationSeasonality", "tree_height", "MeanDiurnalRange",
            "AnnualVapr", "MeanTemperatureofWarmestQuarter", "PrecipitationofWarmestQuarter", 
            "MaxTemperatureofWarmestMonth", "Isothermality"]
X = age_data[feature_].to_array().transpose('plot', 'sample_plot', 'variable').values
Y = age_data['age'].where(age_data['oldgrowth']==0).values

#%% Split train and valid set
n_features = X.shape[2]
train_indx, valid_indx = train_test_split(np.arange(Y.shape[0]), test_size = 0.3)
X_train, Y_train, X_valid, Y_valid = X[train_indx], Y[train_indx], X[valid_indx], Y[valid_indx]
X_train, Y_train, X_valid, Y_valid = X_train.reshape(-1, n_features), Y_train.reshape(-1), X_valid.reshape(-1, n_features), Y_valid.reshape(-1)
mask_train = (np.all(np.isfinite(X_train), axis=1)) & (np.isfinite(Y_train))
X_train, Y_train = X_train[mask_train, :], Y_train[mask_train]
mask_valid = (np.all(np.isfinite(X_valid), axis=1)) & (np.isfinite(Y_valid))
X_valid, Y_valid = X_valid[mask_valid, :], Y_valid[mask_valid]
    
# Fits the model with X and Y
print('Model training regression')
#best_model_reg = hp_tune_reg.tune_(X_train, Y_train, X_valid, Y_valid)
#dump(best_model_reg,'/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_reg.joblib')

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
age_product.to_netcdf('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/age_globbiomass_TC_030_new.nc', encoding={'age': {'dtype': np.float32,
                                                                                                                             'zlib': True, 'complevel': 9}})
