#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:29:09 2023

@author: simon
"""

import pandas as pd 
import numpy as np

FIA_data = pd.read_csv('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/test.csv')
FIA_data = FIA_data.iloc[np.where(FIA_data['agb']<2000)]
FIA_data['age'][FIA_data['age']> 300] =300
FIA_data['agb'] = FIA_data['agb'] * 100
FIA_data = FIA_data.iloc[np.where(FIA_data['age']<300)]
FIA_data = FIA_data.dropna(subset = ['age'])
FIA_data = FIA_data.dropna(subset = ['agb'])
FIA_data['age_class'] = np.round(FIA_data['age'] , -1)
age_counts = FIA_data['age_class'].value_counts()

train_data = pd.read_csv('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/test2.csv')
train_data = train_data[train_data['sourceName'] != 'FIA']
non_OG = train_data.iloc[np.where(train_data['age']<300)]
non_OG['age_class'] = np.round(non_OG['age'], -1)
age_counts = non_OG['age_class'].value_counts()

# Calculate the current number of points per class
current_points_per_class = non_OG['age_class'].value_counts().sort_index()

# Oversample and undersample to balance the dataset
balanced_data_OG = pd.DataFrame()
balanced_data_nonOG = pd.DataFrame()

for age_class, current_points in current_points_per_class.items():
    # Determine the required number of samples to achieve the desired points per class
    desired_points_per_class = 2000
    required_samples = desired_points_per_class - current_points
    
    if required_samples > 0:
        # Oversample by randomly selecting samples with replacement from the current class
        try: 
            oversampled_data = FIA_data[FIA_data['age_class'] == age_class].sample(n=required_samples, replace=False)
        except ValueError:
            oversampled_data = FIA_data[FIA_data['age_class'] == age_class]
        oversampled_data = pd.concat([non_OG[non_OG['age_class'] == age_class], oversampled_data])
        balanced_data_nonOG = pd.concat([balanced_data_nonOG, oversampled_data])
    else:
        balanced_data_nonOG = pd.concat([balanced_data_nonOG, non_OG[non_OG['age_class'] == age_class]])
        


balanced_data= pd.concat([balanced_data_nonOG, balanced_data_OG])

# Add cluster
def round_to_10_degree(coord):
    return np.floor(coord / 10.0) * 10.0

# Apply the functions and combine them to form cluster labels
balanced_data['Latitude_Cluster'] = balanced_data['latitude'].apply(round_to_10_degree)
balanced_data['Longitude_Cluster'] = balanced_data['longitude'].apply(round_to_10_degree)
balanced_data['cluster'] = balanced_data['Latitude_Cluster'].astype(str) + '_' + balanced_data['Longitude_Cluster'].astype(str)
balanced_data.drop(['Latitude_Cluster', 'Longitude_Cluster'], axis=1, inplace=True)
balanced_data['cluster'], _ = pd.factorize(balanced_data['cluster'])

# Export to csv
balanced_data.to_csv('/home/simon/Documents/science/GFZ/projects/forest-age-upscale/data/training_data/training_data_ageMap_OG300_v4.csv')



