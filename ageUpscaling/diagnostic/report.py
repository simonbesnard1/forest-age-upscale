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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import classification_report

from ageUpscaling.utils.metrics import mef_gufunc, nrmse_gufunc, rmse_gufunc

class Report: 

    """
    A class for generating a report on a model experiment.
    
    Attributes:
        study_dir (str): The path to the directory containing the model experiment.
        ds (xarray.Dataset): A dataset containing the model output.
        report_dir (str): The path to the directory where the report will be saved.
        
    """
    
    def __init__(self, 
                 study_dir:str) -> None:
        
        """
        Initializes a new instance of the `Report` class.
        
        Args:
            study_dir (str): The path to the directory containing the model experiment.
            
        """
        
        self.study_dir = study_dir
                
        self.ds = xr.open_zarr(os.path.join(study_dir, 'model_output'))
        self.report_dir = os.path.join(study_dir, 'report')
        
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        
    def xval_diagnostic(self):
        
        obs_ = self.ds.forestAge_obs.values.reshape(-1)
        pred_ = self.ds.forestAge_pred.values.reshape(-1)
        pred_[pred_>299] = 299
        pred_[pred_<0] = 0
        valid_values = np.isfinite(pred_) & np.isfinite(obs_)
        pred_ = pred_[valid_values]
        obs_ = obs_[valid_values]
        
        obs_class = self.ds.oldGrowth_obs.values.reshape(-1)
        pred_class = np.rint(self.ds.oldGrowth_pred.values.reshape(-1))
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
        im = ax[0,0].pcolor(plotMat, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='hot_r', vmin=0.75, vmax=1.0)    
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
        
        im = ax[0,1].hexbin(pred_, obs_, bins='log', gridsize=80, mincnt=2)
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
        fig.delaxes(ax[1,1])
        plt.savefig(os.path.join(self.report_dir, 'xval_diagnostic.png'), dpi=300)        
        plt.close("all")

    def generate_diagnostic(self,
                            diagnostic_type: dict = {'cross-validation'}):
        if 'cross-validation' in diagnostic_type:
            print('Computing cross-validation diagnostic')
            self.xval_diagnostic()
        