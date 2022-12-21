#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:54:54 2022

@author: simon
"""

def interpolate_worlClim(source_ds, 
                         target_ds,
                         method:str = 'linear'):
    
    resampled = source_ds.interp(latitude = target_ds.latitude, 
                                 longitude = target_ds.longitude,
                                 method=method)
    
    if not (resampled.latitude.data == target_ds.latitude.data).all():
        raise ValueError("Failed to interpolate in the latitude axis")
        
    if not (resampled.longitude.data == target_ds.longitude.data).all():
        raise ValueError("Failed to interpolate in the longitude axis")
    
    return resampled
