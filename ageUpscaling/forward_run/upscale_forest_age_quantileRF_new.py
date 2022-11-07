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
from numba import jit, prange, set_num_threads

@jit(nopython=True, parallel=True)
def find_quant(trainy, train_tree_node_ID, pred_tree_node_ID, qntl):
    """find_quant(trainy, train_tree_node_ID, pred_tree_node_ID, qntl)
    
    Aggregates the leaves from the random forest and calculates the quantiles.
    Aggregates leaves based on the tree node indexes from both the training
    and prediction data. Values from the training target data is then used
    to rebuild the leaves for each prediction, which is then summarized
    to the specified quantiles. This is the slowest step in the process,
    so numba is used to speed up this step.
    Parameters
    ----------
    trainy : numpy array of shape (n_target)
        The origianl training target data
    train_tree_node_ID : numpy array of shape (n_training_samples, n_trees)
        array of leaf indices from the training data
    pred_tree_node_ID : numpy array of shape (n_predict_samples, n_trees)
        array of leaf indices from the prediction data
    qntl : numpy array
        quantiles used, must range from 0 to 1
    Returns
    -------
    out : numpy array of shape (n_predict_samples, n_qntl)
        prediction for each quantile
    """
    npred = pred_tree_node_ID.shape[0]
    out = np.zeros((npred, qntl.size))*np.nan
    for i in prange(pred_tree_node_ID.shape[0]):
        idxs = np.where(train_tree_node_ID == pred_tree_node_ID[i, :])[0]
        sample = trainy[idxs]
        out[i, :] = np.quantile(sample, qntl)
    return out


class QuantileRandomForestRegressor:
    """A quantile random forest regressor based on the scikit-learn RandomForestRegressor
    
    A wrapper around the RandomForestRegressor which summarizes based on quantiles rather than
    the mean. Note that quantile predicitons take much longer than mean predictions.
    Parameters
    ----------
    nthreads : int, default=1
        number of threads to used
    rf_kwargs : array or array like
        kwargs to be passed to the RandomForestRegressor
    
    See Also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor.apply
    """
    def __init__(self, nthreads=1, **rf_kwargs):
        rf_kwargs['n_jobs'] = nthreads
        self.forest = RandomForestRegressor(**rf_kwargs)
        set_num_threads(nthreads)

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
        """
        self.forest.fit(X, y, sample_weight)
        self.trainy = y.copy()
        self.trainX = X.copy()

    def predict(self, X, qntl):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        qntl : {array-like} of shape (n_quantiles)
            Quantile or sequence of quantiles to compute, which must be between
            0 and 1 inclusive. Passed to numpy.quantile.
        Returns
        -------
        y : ndarray of shape (n_samples, n_qntl)
            The predicted values.
        """
        qntl = np.asanyarray(qntl)
        ntrees = self.forest.n_estimators
        ntrain = self.trainy.shape[0]
        train_tree_node_ID = np.zeros([ntrain, ntrees])
        npred = X.shape[0]
        pred_tree_node_ID = np.zeros([npred, ntrees])

        for i in range(ntrees):
            train_tree_node_ID[:, i] = self.forest.estimators_[i].apply(self.trainX)
            pred_tree_node_ID[:, i] = self.forest.estimators_[i].apply(X)

        ypred_pcts = find_quant(self.trainy, train_tree_node_ID,
                                pred_tree_node_ID, qntl)

        return ypred_pcts

    def apply(self, X):
        """
        wrapper for sklearn.ensemble.RandomForestRegressor.apply
        Apply trees in the forest to X, return leaf indices.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        return self.forest.apply(X)

    def decision_path(self, X):
        """
        wrapper for sklearn.ensemble.RandomForestRegressor.decision_path
        Return the decision path in the forest.
        .. versionadded:: 0.18
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.
        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        return self.forest.decision_path(X)

    def set_params(self, **params):
        """
        wrapper for sklearn.ensemble.RandomForestRegressor.set_params
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        return self.forestset_params(**params)

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
        qrf = QuantileRandomForestRegressor(nthreads = 1,
                                            n_estimators = best_model.n_estimators,
                                            max_features = best_model.max_features,
                                            max_depth = best_model.max_depth,
                                            min_samples_split = best_model.min_samples_split,
                                            min_samples_leaf = best_model.min_samples_leaf,
                                            bootstrap = best_model.bootstrap)

        #run model
        qrf.fit(X, Y)
        quantile_pred = qrf.predict(X_upscale_reg[mask], [0.25, 0.5, 0.75])
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

#%% Run upscaling
#print('Upscaling procedure')   
nLatChunks = 50
nLonChunks = 2
LatChunks  = np.linspace(90,-90,nLatChunks)
LonChunks  = np.linspace(-180,180,nLonChunks)
AllExtents = []
for lat in range(nLatChunks-1):
    for lon in range(nLonChunks-1):
        AllExtents.append({'latitude':slice(LatChunks[lat],LatChunks[lat+1]),'longitude':slice(LonChunks[lon],LonChunks[lon+1])})
njobs = 5
#p=mp.Pool(njobs,maxtasksperchild=1)
#p.map(Forward,AllExtents)
#p.close()
#p.join()

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

## Compute IQR
age_productIQR = (age_productQ75 - age_productQ25)

# Create xarray and export ndcf file
TempAnnualRange = xr.open_dataset("/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WorldClim/v2/Data/TemperatureAnnualRange.nc")
age_product = xr.Dataset(data_vars={'age_median': (('latitude', 'longitude'), age_productQ50),
                                    'age_IQR': (('latitude', 'longitude'), age_productIQR),                             
                                    'age_q25': (('latitude', 'longitude'), age_productQ25),
                                    'age_q75':  (('latitude', 'longitude'), age_productQ75)},
                        coords={'latitude': TempAnnualRange.coords["latitude"],
                                'longitude': TempAnnualRange.coords["longitude"]})
## Mask old-growth
NOG = xr.open_dataset('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/classProbaNoOG_globbiomass_TC_010_RF.nc').proba_nonoldgrowth
age_product = age_product.where(np.isfinite(NOG))
age_product.to_netcdf('/Net/Groups/BGI/work_2/FIDC_age_upscale/output/upscale_product/original_product/age_globbiomass_TC_010_quantileRF.nc',
                         encoding={'age_median': {'dtype': np.float32, 'zlib': True, 'complevel': 9},
                                   'age_IQR': {'dtype': np.float32, 'zlib': True, 'complevel': 9},
                                   'age_q25': {'dtype': np.float32, 'zlib': True, 'complevel': 9},
                                   'age_q75': {'dtype': np.float32, 'zlib': True, 'complevel': 9}})
