#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
import os

def id_get_result(id, ion):
    
    if ion == 'Li':
        predictions = pd.read_csv('predict_data/vol_pred_Li.csv')
    elif ion=='Na':
        predictions = pd.read_csv('predict_data/vol_pred_Na.csv')
    elif ion=='K':
        predictions = pd.read_csv('predict_data/vol_pred_K.csv')
    elif ion=='Rb':
        predictions = pd.read_csv('predict_data/vol_pred_Rb.csv')
    elif ion=='Cs':
        predictions = pd.read_csv('predict_data/vol_pred_Cs.csv')
    elif ion=='Mg':
        predictions = pd.read_csv('predict_data/vol_pred_Mg.csv')
    elif ion=='Ca':
        predictions = pd.read_csv('predict_data/vol_pred_Ca.csv')
    elif ion=='Zn':
        predictions = pd.read_csv('predict_data/vol_pred_Zn.csv')
    elif ion=='Y':
        predictions = pd.read_csv('predict_data/vol_pred_Y.csv')
    elif ion=='Al':
        predictions = pd.read_csv('predict_data/vol_pred_Al.csv')
    else:
        raise ValueError('invalid working_ion')
        
    
    if not predictions[predictions['entry_id'] == id].empty:
            result = predictions[predictions['entry_id'] == id].values[0].tolist()
    else:
        raise ValueError('invalid id')
    return result

