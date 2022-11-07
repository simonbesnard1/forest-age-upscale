#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:23:17 2019

@author: sbesnard
"""
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import multiprocessing as mp
import xarray as xr
from joblib import dump, load
import os

class RFP:
    def __init__(self, trainRFy, TreeNodeIDXTrain,TreeNodeIDXPred,pcts):
        self.trainRFy=trainRFy
        self.TreeNodeIDXTrain=TreeNodeIDXTrain
        self.TreeNodeIDXPred=TreeNodeIDXPred
        self.pcts=pcts
    def Calc_pcts(self,i):
        return(np.percentile(self.trainRFy[np.where(self.TreeNodeIDXTrain==self.TreeNodeIDXPred[i,:])[0]],self.pcts))

def RFPercentilePrediction(Forest,trainRFxs,trainRFy,predRFxs,pcts=[5,50,95],n_jobs=1,maxtasksperchild=500):
    """RFQuantilePrediction(Forest,trainRFxs,trainRFy,predRFxs,pcts=[5,50,95])

    Random Forest Percentile Prediction

    Fits a slope between x and y individually with an intercept of zero, then takes the geometric mean of the slopes.


    Parameters
    ----------
    Forest : class 'sklearn.ensemble.forest.RandomForestRegressor'
        Fitted random forest regressor from sklearn
    trainRFxs : numpy array
        The x dataset used to fit Forest
    trainRFy : numpy array
        The y dataset used to fit Forest
    predRFxs : numpy array
        The x dataset to get the percentile prediction
    pcts : list or list like
        The percentiles to output, defaults are the 5th, 50th, and 95th percentiles
    n_jobs:
        Passed to multiprocessing.Pool number of processors to use.
    maxtasksperchild : integer
        Passed to multiprocessing.Pool job tasts to complete before being cleaned, to save memory.

     Returns
    -------
    numpy ndarray
        The resulting percentile prediction shaped [npred,npcts]
    """

    ntrees=Forest.n_estimators
    n=trainRFy.shape[0]
    TreeNodeIDXTrain=np.zeros([n,ntrees])
    npred=predRFxs.shape[0]
    TreeNodeIDXPred=np.zeros([npred,ntrees])


    for i in range(ntrees):
        TreeNodeIDXTrain[:,i]=Forest.estimators_[i].apply(trainRFxs)
        TreeNodeIDXPred[:,i]=Forest.estimators_[i].apply(predRFxs)

    ypred_pcts=np.ones([npred,len(pcts)])*np.nan

    c=RFP(trainRFy, TreeNodeIDXTrain,TreeNodeIDXPred,pcts)

    if n_jobs>1:
        with mp.Pool(processes=n_jobs,maxtasksperchild=maxtasksperchild) as p:
            ypred_pcts=np.array(list(p.map(c.Calc_pcts, range(npred))))
    else:
        ypred_pcts=np.array(list(map(c.Calc_pcts, range(npred))))

    return(ypred_pcts)

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
    feature_reg = np.concatenate(pd.read_csv('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/feature_selection/variable_selected_borutaReg.csv').values)
    X_upscale_reg = data_cube[feature_reg].to_array().transpose('latitude', 'longitude', 'variable').values.reshape(-1,feature_reg.shape[0])
    Q25_pred, Q50_pred, Q75_pred = np.zeros(X_upscale_reg.shape[0]) * np.nan, np.zeros(X_upscale_reg.shape[0]) * np.nan, np.zeros(X_upscale_reg.shape[0]) * np.nan
    mask = np.all(np.isfinite(X_upscale_reg), axis=1)
    if (X_upscale_reg[mask].shape[0]>0):
        #load model
        best_model= load('/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_RFregression.joblib')        
        #run model
        quantile_pred = RFPercentilePrediction(best_model, X, Y, X_upscale_reg[mask], pcts = [25,50,75],n_jobs=1)
        Q25_pred[mask], Q50_pred[mask], Q75_pred[mask] = quantile_pred[:,0], quantile_pred[:,1], quantile_pred[:,2]
    Q25_pred, Q50_pred,Q75_pred = Q25_pred.reshape(OrigShape), Q50_pred.reshape(OrigShape), Q75_pred.reshape(OrigShape) 
    #Output the data to numpy binary files to be loaded later
    FileName = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q25/Lat{0}x{1}_Lon{2}x{3}.npy'.format(extents['latitude'].start,extents['latitude'].stop,extents['longitude'].start,extents['longitude'].stop)
    np.save(FileName,Q25_pred)
    FileName = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q50/Lat{0}x{1}_Lon{2}x{3}.npy'.format(extents['latitude'].start,extents['latitude'].stop,extents['longitude'].start,extents['longitude'].stop)
    np.save(FileName,Q50_pred)
    FileName = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q75/Lat{0}x{1}_Lon{2}x{3}.npy'.format(extents['latitude'].start,extents['latitude'].stop,extents['longitude'].start,extents['longitude'].stop)
    np.save(FileName,Q75_pred)
 
#%% Load forest age data    
print('Loading FIDC data')      
age_data = xr.open_dataset("/Net/Groups/BGI/work_2/FIDC_age_upscale/input/training_data/training_data_ageMap_OG300_new.nc")

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
print('Model training regression')
#best_model_reg = RFregression.tune_(X, Y)
#dump(best_model_reg,'/Net/Groups/BGI/work_2/FIDC_age_upscale/model/best_model_RFregression.joblib')

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

# Load numpy arrays and stack them - Q25
print('Combining chuncks and creating final product')   
Q5_pred_dir = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q25'
fileList= os.listdir(Q5_pred_dir)
for extents in range(len(AllExtents)):
    fileList[extents] = np.load('/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q25/Lat{0}x{1}_Lon{2}x{3}.npy'.format(AllExtents[extents]['latitude'].start,AllExtents[extents]['latitude'].stop,AllExtents[extents]['longitude'].start,AllExtents[extents]['longitude'].stop)).reshape(-1)
age_productQ25 = np.concatenate(fileList)
age_productQ25 = np.array(age_productQ25.reshape(21600, 43200))

# Load numpy arrays and stack them - Q50
Q50_pred_dir = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q50'
fileList= os.listdir(Q50_pred_dir)
for extents in range(len(AllExtents)):
    fileList[extents] = np.load('/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q50/Lat{0}x{1}_Lon{2}x{3}.npy'.format(AllExtents[extents]['latitude'].start,AllExtents[extents]['latitude'].stop,AllExtents[extents]['longitude'].start,AllExtents[extents]['longitude'].stop)).reshape(-1)
age_productQ50 = np.concatenate(fileList)
age_productQ50 = np.array(age_productQ50.reshape(21600, 43200))

# Load numpy arrays and stack them - Q95
Q95_pred_dir = '/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q75'
fileList= os.listdir(Q95_pred_dir)
for extents in range(len(AllExtents)):
    fileList[extents] = np.load('/Net/Groups/BGI/scratch/sbesnard/age_upscale/np_chunck/Q75/Lat{0}x{1}_Lon{2}x{3}.npy'.format(AllExtents[extents]['latitude'].start,AllExtents[extents]['latitude'].stop,AllExtents[extents]['longitude'].start,AllExtents[extents]['longitude'].stop)).reshape(-1)
age_productQ75 = np.concatenate(fileList)
age_productQ75 = np.array(age_productQ75.reshape(21600, 43200))

# Compute meand standard deviation
age_productMean = (age_productQ50 - age_productQ25) / 2
age_productSTD = (age_productQ75 - age_productMean) / 1.645
age_productIQR = (age_productQ75 - age_productQ25)

# Create xarray and export ndcf file
TempAnnualRange = xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/TemperatureAnnualRange.nc")
age_product = xr.Dataset(data_vars={'age_mean':    (('latitude', 'longitude'), age_productMean),
                             'age_std':    (('latitude', 'longitude'), age_productSTD),
                             'age_median': (('latitude', 'longitude'), age_productQ50),
                             'age_IQR': (('latitude', 'longitude'), age_productIQR),                             
                             'age_q25': (('latitude', 'longitude'), age_productQ25),
                             'age_q75':  (('latitude', 'longitude'), age_productQ75)},
        coords={'latitude': TempAnnualRange.coords["latitude"],
                'longitude': TempAnnualRange.coords["longitude"]})
age_product.to_netcdf('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/age_globbiomass_TC_010_quantileRF.nc')
