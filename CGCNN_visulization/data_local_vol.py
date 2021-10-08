#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load data.py
from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
#from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


def CGCNNdata(Dataset, normalizer, model): 
    #atom_tablenum=[]
    #pre_out=[]
    #cif_ids=[]
    #atom_bond=[]
    #atom_num=[]
    #target_vol=[]
    cgcnn_dataset=[]
    for i, (input, target, batch_cif_ids, atom_table_num) in enumerate(Dataset):
        vol_out=[]
        with torch.no_grad():
            input_var = (Variable(input[0]),
                        Variable(input[1]),
                        input[2],
                        input[3])
        #atom_bond.append(input[2])
        #atom_num.append(input[3])
        target_normed = normalizer.norm(target)
        with torch.no_grad():
            target_var = Variable(target_normed)

        output=model(*input_var)

        #local_voltage=LocalEnergy(output[1][1])

        vol_out.append(output)
        vol_out.append(input[2])
        vol_out.append(input[3])
        vol_out.append(torch.reshape(atom_table_num, [-1]))
        vol_out.append(batch_cif_ids)
        vol_out.append(target[0])
        cgcnn_dataset.append(vol_out)
        #atom_tablenum.append(torch.reshape(atom_table_num, [-1]))
        #cif_ids.append(batch_cif_ids)
        #target_vol.append(target)

        '''
        pre_out[0] the predicted voltage values.
        pre_out[1] the output of every layer
        pre_out[1][0] embedding
        pre_out[1][1] conv
        pre_out[1][2] pooling
        pre_out[1][3] hidden layer
        '''
    return cgcnn_dataset #pre_out, atom_tablenum, cif_ids, atom_bond, atom_num, target_vol

def collate_pool_local(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    pre_out, atom_tablenum, cif_ids, atom_bond, atom_num, target_vol
    """
    batch_nbr_fea= []
    
    bond_atom_incrystal, batch_atom_table_num =[], []
    batch_nbr_fea_idx, crystal_atom_idx=[], []
    batch_cif_ids=[]
    batch_target=[]  

    base_idx = 0

    for i, data_sample in enumerate(dataset_list):
        
        pre_out=data_sample[0]
        nbr_fea=pre_out[1][1]
        batch_nbr_fea.append(nbr_fea)
        
        n_i = nbr_fea.shape[0]
        batch_nbr_fea_idx.append(data_sample[1]+base_idx)
        #crystal_atom_idx.append(data_sample[2]+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        base_idx += n_i
        
        batch_atom_table_num.append(data_sample[3])
        batch_cif_ids.append(data_sample[4])
        batch_target.append(data_sample[5])
        
  
        
    return (torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids,\
        torch.cat(batch_atom_table_num, dim=0)