#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sbesnard
@File    :   report.py
@Time    :   Mon Sep 26 10:47:17 2022
@Author  :   Simon Besnard
@Version :   1.0
@Contact :   besnard@gfz-potsdam.de
@License :   (C)Copyright 2022-2023, GFZ-Potsdam
@Desc    :   A method class for creating diagnostic plots.
"""

import os

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as st

from sklearn.metrics import classification_report

from ageUpscaling.utils.metrics import mef_gufunc, nrmse_gufunc, rmse_gufunc

def violins(data,pos=0,bw_method=None,resolution=50,spread=1,max_num_points=100):
    """violins(data,pos=0,bw_method=None,resolution=50,spread=1)
    Jitter violin plot creater
    Takes points from a distribution and creates data for both a jitter violin and a standard violin plot.
    Parameters
    ----------
    data : numpy array
        The data to build the violin plots from
    pos : float or int
        The position the resulting violin will be centered on
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth. This can be ‘scott’, ‘silverman’, a scalar constant or a callable. If a scalar, this will be used directly as kde.factor. If a callable, it should take a gaussian_kde instance as only parameter and return a scalar. If None (default), ‘scott’ is used. See Notes for more details.
    resolution : int
        The resolution of the resulting violin plot
    spread : int or float
        The spread of the violin plots
     Returns
    -------
    pointx,pointy : numpy arrays
        The resulting data for the jitter violin plot (use with pl.scatter)
    fillx,filly : numpy array
        The resulting data for a standard violin plot (use with pl.fill_between)
    """
    if data.size>max_num_points:
        data = np.random.choice(data,size=max_num_points,replace=False)
    kde    = st.gaussian_kde(data,bw_method=bw_method)
    pointx = data
    pointy = kde.pdf(pointx)
    pointy = pointy/(2*pointy.max())
    fillx  = np.linspace(data.min(),data.max(),resolution)
    filly  = kde.pdf(fillx)
    filly  = filly/(2*filly.max())
    pointy = pos+np.where(np.random.rand(pointx.shape[0])>0.5,-1,1)*np.random.rand(pointx.shape[0])*pointy*spread
    filly  = (pos-filly*spread,pos+filly*spread)
    return(pointx,pointy,fillx,filly)

class Report: 

    """
    A class for generating a report on a model experiment.
    
    Attributes:
        study_dir (str): The path to the directory containing the model experiment.
        ds (xarray.Dataset): A dataset containing the model output.
        report_dir (str): The path to the directory where the report will be saved.
        
    """
    
    def __init__(self, 
                 study_dir:str=None,
                 nfi_data:str = None,
                 dist_cube:str = None) -> None:
        
        """
        Initializes a new instance of the `Report` class.
        
        Args:
            study_dir (str): The path to the directory containing the model experiment.
            
        """
        
        self.study_dir = study_dir
        self.nfi_data = nfi_data
        self.dist_cube = dist_cube
        self.report_dir = os.path.join(study_dir, 'report')
        
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        
    def xval_diagnostic(self):
        
        ds = xr.open_zarr(os.path.join(self.study_dir, 'model_prediction'))
        
        obs_ = ds.forestAge_obs.values.reshape(-1)
        pred_ = ds.forestAge_pred.values.reshape(-1)
        pred_[pred_>299] = 299
        pred_[pred_<1] = 1
        obs_[obs_>=300] = np.nan        
        valid_values = np.isfinite(pred_) & np.isfinite(obs_)
        pred_ = pred_[valid_values]
        obs_ = obs_[valid_values]
        
        obs_class = ds.oldGrowth_obs.values.reshape(-1)
        pred_class = ds.oldGrowth_pred.values.reshape(-1)
        valid_values = np.isfinite(pred_class) & np.isfinite(obs_class)
        pred_class = pred_class[valid_values]
        obs_class = obs_class[valid_values]        
        cr_ = classification_report(obs_class, pred_class, labels = [0,1], target_names=['non old-growth', 'old-growth'])
        lines = cr_.split('\n')[2:]
        classes = []
        plotMat = []
        support = []
        class_names = []
        for line in lines[: 2]:
            t = list(filter(None, line.strip().split('  ')))
            if len(t) < 4: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            plotMat.append(v)

        fig, ax = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={'wspace': 0.35, 'hspace': 0.35})
        im = ax[0,0].pcolor(plotMat, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='hot_r', vmin=np.min(plotMat)-0.05, vmax=np.max(plotMat))    
        ax[0,0].set_yticks(np.arange(2) + 0.5, minor=False)
        ax[0,0].set_xticks(np.arange(3) + 0.5, minor=False)
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
        ax[0,0].set_xticklabels(xticklabels, minor=False)
        ax[0,0].set_yticklabels(yticklabels, minor=False, rotation=40)
        divider = make_axes_locatable(ax[0,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax[0,0].text(0.05, 0.95, "A", transform=ax[0,0].transAxes,
                fontsize=16, fontweight='bold', va='top')
        
        im = ax[0,1].hexbin(pred_, obs_, bins='log', gridsize=100, mincnt=2)
        ax[0,1].set_xlabel('Predicted forest age [years]', size=12)   
        ax[0,1].set_ylabel('Observed forest age [years]', size=12)
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)
        ax[0,1].text(0.05, 0.95, "B", transform=ax[0,1].transAxes,
                fontsize=16, fontweight='bold', va='top')
        fig.colorbar(im, ax=ax[0,1])
        ax[0,1].set_ylim(0,310)
        ax[0,1].set_xlim(0,310)
        slope_, intercept_ = np.polyfit(pred_, obs_, 1)
        ax[0,1].plot(pred_, slope_*pred_ + intercept_, color='black', linewidth = 2, linestyle='dashed')
        r = mef_gufunc(pred_, obs_)
        rmse = rmse_gufunc(pred_, obs_)
        nrmse = nrmse_gufunc(pred_, obs_) *100
        text_box = dict(facecolor='white', edgecolor='none', pad=4, alpha=.9)
        title=''
        ax[0,1].text(0.55, 0.24, f'{title}  NSE={r:.2f}', horizontalalignment='left',
            verticalalignment='top', transform=ax[0,1].transAxes, bbox=text_box, size=12)  
        ax[0,1].text(0.55, 0.16, f'{title}  RMSE={rmse:.2f} yr', horizontalalignment='left',
            verticalalignment='top', transform=ax[0,1].transAxes, bbox=text_box, size=12)
        ax[0,1].text(0.55, 0.08, f'{title}  NRMSE={nrmse:.2f} %', horizontalalignment='left',
            verticalalignment='top', transform=ax[0,1].transAxes, bbox=text_box, size=12)
        ax[0,1].plot([0, 310], [0, 310], linestyle='--', color='red', linewidth=2)
          
        percs = np.linspace(0,100,1000)
        qn_pred = np.percentile(pred_, percs)
        qn_obs = np.percentile(obs_, percs)
        ax[1,0].scatter(qn_pred,qn_obs, marker="o", s=10)
        ax[1,0].set_xlabel('Predicted forest age [years]', size=12)   
        ax[1,0].set_ylabel('Observed forest age [years]', size=12)
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)
        ax[1,0].text(0.05, 0.95, "C", transform=ax[1,0].transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax[1,0].set_ylim(0,310)
        ax[1,0].set_xlim(0,310)
        ax[1,0].plot([0, 310], [0, 310], linestyle='--', color='red', linewidth=2)
        
        residual = obs_ - pred_
        age_classes = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100), (100, 150), (150, 200), (200, 300)]
        for j in range(len(age_classes)):
            
            Agemask = (obs_ >= age_classes[j][0]) & (obs_ < age_classes[j][1])
            
            residual_masked = residual[Agemask]
            IQ_mask = (residual_masked > np.quantile(residual_masked, 0.25)) & (residual_masked < np.quantile(residual_masked, 0.75))
            pointx,pointy,fillx,filly = violins(residual_masked[IQ_mask],pos=j,spread=0.3,max_num_points=1000)
            ax[1,1].scatter(pointy, pointx, color='darkblue',alpha=0.1, marker='.') 
            ax[1,1].scatter(j, np.nanquantile(residual_masked[IQ_mask], 0.5), marker='d', s=200, color='red', alpha=0.5)
    
        # Extract age class labels for x-axis
        age_class_labels = [f'{lower}-{upper}' for (lower, upper) in age_classes]
        
        # Create the bar plot
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)
        ax[1,1].set_ylabel('Model residuals [years]', size=12)
        ax[1,1].text(0.05, 0.95, "D", transform=ax[1,1].transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax[1,1].set_xticks(range(len(age_classes)))
        ax[1,1].set_ylim(-50, 120)
        ax[1, 1].axhline(y=0, color='black', linestyle='--', linewidth =3)
        ax[1,1].text(-0.3, 8, 'Perfect model', color='black', fontsize=12)       
        ax[1,1].set_xticklabels(age_class_labels, rotation=45)
        
        plt.savefig(os.path.join(self.report_dir, 'xval_diagnostic.png'), dpi=300)        
        plt.close("all")
        
    def GlobalAge_diagnostic(self):
        
        ds = xr.open_zarr(os.path.join(self.study_dir, 'AgeUpscale_100m')).isel(time=1).forest_age
        
        fig,ax = plt.subplots(ncols=2,nrows=3,figsize=(6,7), constrained_layout = True)

        dat_= ds.sel(latitude = slice(6, 2), longitude = slice(-65, -61))
        dat_.attrs['long_name'] = 'Forest Age'
        dat_.plot.imshow(ax=ax[0,0], cmap= "afmhot_r", vmin=0, vmax =150,
                         cbar_kwargs = dict(orientation='vertical', shrink=0.7, aspect=10, pad=0.05))
        ax[0,0].set_title("Amazon basin")

        dat_= ds.sel(latitude = slice(-1, -5), longitude = slice(19, 23))
        dat_.attrs['long_name'] = 'Forest Age'
        dat_.plot.imshow(ax=ax[0,1], cmap= "afmhot_r", vmin=0, vmax =150,
                             cbar_kwargs = dict(orientation='vertical', shrink=0.6, aspect=10, pad=0.05))
        ax[0,1].set_title("Congo basin")

        dat_= ds.sel(latitude = slice(45, 41), longitude = slice(0, 4))
        dat_.attrs['long_name'] = 'Forest Age'
        dat_.plot.imshow(ax=ax[1,0], cmap= "afmhot_r", vmin=0, vmax =150,
                             cbar_kwargs = dict(orientation='vertical', shrink=0.6, aspect=10, pad=0.05))
        ax[1,0].set_title("Europe")

        dat_= ds.sel(latitude = slice(66, 62), longitude = slice(49, 53))
        dat_.attrs['long_name'] = 'Forest Age'
        dat_.plot.imshow(ax=ax[1,1], cmap= "afmhot_r", vmin=0, vmax =150,
                             cbar_kwargs = dict(orientation='vertical', shrink=0.7, aspect=10, pad=0.05))
        ax[1,1].set_title("Siberia")

        dat_= ds.sel(latitude = slice(55, 51), longitude = slice(-114, -110))
        dat_.attrs['long_name'] = 'Forest Age'
        dat_.plot.imshow(ax=ax[2,0], cmap= "afmhot_r", vmin=0,vmax =150,
                             cbar_kwargs = dict(orientation='vertical', shrink=0.6, aspect=10, pad=0.05))
        ax[2,0].set_title("North America")

        dat_= ds.sel(latitude = slice(31, 27), longitude = slice(109, 113))
        dat_.attrs['long_name'] = 'Forest Age'
        dat_.plot.imshow(ax=ax[2,1], cmap= "afmhot_r", vmin=0, vmax =150,
                             cbar_kwargs = dict(orientation='vertical', shrink=0.6, aspect=10, pad=0.05))
        ax[2,1].set_title("China")
        plt.savefig(os.path.join(self.report_dir, 'GlobalAge_diagnostic.png'), dpi=300)        
        plt.close("all")

    def NFI_diagnostic(self):
        
        #%% Extract predicted age for nfi data
        global_age = xr.open_zarr(os.path.join(self.study_dir, 'AgeUpscale_100m'))
        dist_data = xr.open_zarr(self.dist_cube)
        nfi_data = pd.read_csv(self.nfi_data)
        nfi_data['year_of_measurement'] = np.round(nfi_data['year_of_measurement'].values).astype('int16')
        nfi_data = nfi_data[nfi_data['year_of_measurement'] >= 2001]
        nfi_data['age'] = np.round(nfi_data['age'].values).astype('int16')
        nfi_data['age'] = nfi_data['age'] + (2020 - nfi_data['year_of_measurement']) 
        nfi_data['age'][nfi_data['age']> 300] = 300
        nfi_data = nfi_data.dropna()
        nfi_data.reset_index(inplace=True)

        age_extract = []
        for index, row in nfi_data.iterrows():
            dist_extract = dist_data.LandsatDisturbanceTime.sel(latitude = row['latitude_origin'], longitude = row['longitude_origin'], method = 'nearest').values
            if dist_extract > 20:
                age_extract.append(global_age.forest_age.sel(time = '2020-01-01', latitude = row['latitude_origin'], longitude = row['longitude_origin'], method = 'nearest').values)
        extracted_df = pd.DataFrame(age_extract, columns=['forest_age'])
        nfi_data = pd.concat([nfi_data, extracted_df], axis=1)
        nfi_data = nfi_data.dropna()
        
        #%% Plot data
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout= True)
        
        obs_ = nfi_data['age'].values.astype("float32")
        pred_ = nfi_data['forest_age'].values.astype("float32")
        valid_values = np.isfinite(pred_) & np.isfinite(obs_)
        pred_ = pred_[valid_values]
        obs_ = obs_[valid_values]
        im = ax[0].hexbin(pred_,obs_, bins='log', gridsize=100, mincnt=3)
        ax[0].set_xlabel('GAMI v2.0 forest age [years]', size=12)   
        ax[0].set_ylabel('NFI forest age [years]', size=12)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        fig.colorbar(im, ax=ax[0])
        ax[0].set_ylim(0,310)
        ax[0].set_xlim(0,310)
        slope_, intercept_ = np.polyfit(pred_, obs_, 1)
        ax[0].plot(pred_, slope_*pred_ + intercept_, color='black', linewidth = 2, linestyle='dashed')
        rmse = rmse_gufunc(pred_, obs_)
        nrmse = nrmse_gufunc(pred_, obs_) *100
        text_box = dict(facecolor='white', edgecolor='none', pad=4, alpha=.9)
        title=''
        ax[0].text(0.55, 0.16, f'{title}  RMSE={rmse:.2f} yr', horizontalalignment='left',
            verticalalignment='top', transform=ax[0].transAxes, bbox=text_box, size=12)
        ax[0].text(0.55, 0.08, f'{title}  NRMSE={nrmse:.2f} %', horizontalalignment='left',
            verticalalignment='top', transform=ax[0].transAxes, bbox=text_box, size=12)
        ax[0].plot([0, 310], [0, 310], linestyle='--', color='red', linewidth=2)
        ax[0].text(0.05, 0.95, "a", transform=ax[0].transAxes,
                fontsize=16, fontweight='bold', va='top')

        residual = obs_ - pred_
        age_classes = [(0, 20), (20, 40), (40, 60), (60, 100), (100, 150), (150, 200), (200, 300)]
        for j in range(len(age_classes)):
            
            Agemask = (pred_ >= age_classes[j][0]) & (pred_ < age_classes[j][1])
            
            residual_masked = residual[Agemask]
            IQ_mask = (residual_masked > np.quantile(residual_masked, 0.25)) & (residual_masked < np.quantile(residual_masked, 0.75))
            pointx,pointy,fillx,filly = violins(residual_masked[IQ_mask],pos=j,spread=0.3,max_num_points=1000)
            ax[1].scatter(pointy, pointx, color='darkblue',alpha=0.1, marker='.') 
            ax[1].scatter(j, np.nanquantile(residual_masked[IQ_mask], 0.5), marker='d', s=200, color='red', alpha=0.5)

        # Extract age class labels for x-axis
        age_class_labels = [f'{lower}-{upper}' for (lower, upper) in age_classes]

        # Create the bar plot
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_ylabel('Age difference [years]', size=12)
        ax[1].text(0.05, 0.95, "b", transform=ax[1].transAxes,
                fontsize=16, fontweight='bold', va='top')
        ax[1].set_xticks(range(len(age_classes)))
        ax[1].axhline(y=0, color='black', linestyle='--', linewidth =3)
        ax[1].set_xticklabels(age_class_labels, rotation=45)
        plt.savefig(os.path.join(self.report_dir, 'nfi_validation.png'), dpi=300)        
        
        plt.close("all")   

    def generate_diagnostic(self,
                            diagnostic_type: dict = {'cross-validation'}):
        if 'cross-validation' in diagnostic_type:
            print('Computing cross-validation diagnostic')
            self.xval_diagnostic()
        if 'global-age' in diagnostic_type:
            print('Computing global forest age diagnostic')
            self.GlobalAge_diagnostic()    
        if 'nfi-valid' in diagnostic_type:
            print('Computing validation with NFI data')
            self.NFI_diagnostic()
            
        
        
