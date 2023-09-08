#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:30:12 2023

@author: simon
"""
import pickle 
import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt

dat_ = pd.read_csv('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_subsetFIA_v6.csv')

with open('/home/simon/Documents/best_Regressor_run0.pickle', 'rb') as f:
    regressor_config = pickle.load(f)    

dat_ = dat_[dat_['age'] < 300]
X = dat_[regressor_config['selected_features']].to_numpy()
mask_nan = np.all(np.isfinite(X), axis=1)
X = X[mask_nan, :]

explainer = shap.TreeExplainer(regressor_config['best_model'])
Xd = xgboost.DMatrix(X)
shap_values = explainer.shap_values(Xd)

#%% Plot importance value
shap.summary_plot(shap_values,features= X, feature_names=regressor_config['selected_features'], show=False)
plt.savefig('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/output/plot/SHAP_values.png', dpi=300)

#%% Plot emerging relation
varname = regressor_config['selected_features']
fig, ax = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
for var_ in range(len(varname)):
    x = X[:, var_]
    y = shap_values[:, var_]
    mask_ = (x < np.nanquantile(x, .99)) & (x > np.nanquantile(x, .01))
    if varname[var_] == 'AnnualVapr_WorlClim':
        im = ax[0,0].hexbin(x[mask_], y[mask_], bins='log', gridsize=120, mincnt=1)
        ax[0,0].set_xlabel('vapr [hPa]', size=14)
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)
        ax[0,0].text(0.05, 1.05, 'A', transform=ax[0,0].transAxes,
                fontsize=16, fontweight='bold', va='top')
        fig.colorbar(im, ax=ax[0,0])
    if varname[var_] == 'agb_gapfilled':
        im= ax[0,1].hexbin(x[mask_], y[mask_], bins='log', gridsize=120, mincnt=1)
        ax[0,1].set_xlabel('AGB [$Mg \ ha^{-1}$]', size=14)
        ax[0,1].spines['top'].set_visible(False)
        ax[0,1].spines['right'].set_visible(False)
        ax[0,1].text(0.05, 1.05, 'B', transform=ax[0,1].transAxes,
                fontsize=16, fontweight='bold', va='top')
        fig.colorbar(im, ax=ax[0,1])
    if varname[var_] == 'MeanTemperatureofDriestQuarter_WorlClim':
        ax[1,0].hexbin(x[mask_], y[mask_], bins='log', gridsize=120, mincnt=1)
        ax[1,0].set_xlabel('MeanTemperatureofDriestQuarter_WorlClim [degC]', size=14)
        ax[1,0].spines['top'].set_visible(False)
        ax[1,0].spines['right'].set_visible(False)
        ax[1,0].text(0.05, 1.05, 'C', transform=ax[1,0].transAxes,
                fontsize=16, fontweight='bold', va='top')
        fig.colorbar(im, ax=ax[1,0])
    if varname[var_] == 'canopy_height_gapfilled':
        ax[1,1].hexbin(x[mask_], y[mask_], bins='log', gridsize=120, mincnt=1)
        #ax[11].set_ylabel('Shap value [$MgC \ ha^{-1} \ year^{-1}$]', size=12)   
        ax[1,1].set_xlabel('canopy_height_gapfilled [meter]', size=14)
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['right'].set_visible(False)
        ax[1,1].text(0.05, 1.05, 'D', transform=ax[1,1].transAxes,
                fontsize=16, fontweight='bold', va='top')
        fig.colorbar(im, ax=ax[1,1])
    if varname[var_] == 'AnnualSrad_WorlClim':
        ax[2,0].hexbin(x, y, bins='log', gridsize=120, mincnt=1)
        ax[2,0].set_xlabel('AnnualSrad_WorlClim [$W \ m^{-2}$]', size=14)
        ax[2,0].spines['top'].set_visible(False)
        ax[2,0].spines['right'].set_visible(False)
        ax[2,0].text(0.05, 1.05, 'E', transform=ax[2,0].transAxes,
                fontsize=16, fontweight='bold', va='top')    
        fig.colorbar(im, ax=ax[2,0])
    if varname[var_] == 'Isothermality_WorlClim':
        ax[2,1].hexbin(x[mask_], y[mask_], bins='log', gridsize=120, mincnt=1)
        ax[2,1].set_xlabel('Isothermality [-]', size=12)
        ax[2,1].spines['top'].set_visible(False)
        ax[2,1].spines['right'].set_visible(False)
        ax[2,1].text(0.05, 1.05, "F", transform=ax[2,1].transAxes,
                fontsize=16, fontweight='bold', va='top')     
        fig.colorbar(im, ax=ax[2,1])
ax[1,0].set_ylabel('Shap values [years]', size=14)        
plt.savefig('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/output/plot/emergent_relationship.png', dpi=300)

