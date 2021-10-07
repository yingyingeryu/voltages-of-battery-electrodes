#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
from cgcnn.local_voltage import LocalEnergy
from cgcnn.main_local_vol import  Normalizer
from cgcnn.data_local_vol import collate_pool_local
from torch.utils.data import DataLoader
from cgcnn.data_local_vol import CGCNNdata
from cgcnn.data_pre import CIFData
from cgcnn.data_pre import collate_pool
from cgcnn.model_pre import CrystalGraphConvNet

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import time
from torch.autograd import Variable


# In[ ]:


modelpath='../cgcnn/model_best.pth.tar'
dataset = CIFData('../../Li_battery_voltage/root_dir_2std_2190')
collate_fn = collate_pool
test_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=0, collate_fn=collate_fn,
                        pin_memory=False)


# In[ ]:


structures, _, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]
model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                            atom_fea_len=64,
                            n_conv=3,
                            h_fea_len=128,
                            n_h=1,
                            classification=False)


# In[ ]:


checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])


# In[ ]:


normalizer = Normalizer(torch.zeros(3))

normalizer.load_state_dict(checkpoint['normalizer'])
best_mae_error = 1e10


# In[ ]:


dataset_CGCNN=CGCNNdata(test_loader, normalizer, model)


# In[ ]:


collate_fn=collate_pool_local

test_loader = DataLoader(dataset_CGCNN, batch_size=1, shuffle=True,
                        num_workers=0, collate_fn=collate_fn,
                        pin_memory=False)


# In[ ]:


atom_bond_fea_len = 64
model_local = LocalEnergy(atom_bond_fea_len)


# In[ ]:


best_checkpoint = torch.load('model_best_cov.pth.tar')
model_local.load_state_dict(best_checkpoint['state_dict'])


# In[ ]:


pre_target, target_vol=[], []
local_energy, atom_tablenumber=[], []
local_cif_ids=[]

for i, ((nbr_fea, atom_bond_num, atom_num), target, batch_cif_ids, atom_table_num) in enumerate(test_loader):
    
    target_vol.append(target.cpu().item())
    
    local_cif_ids.append(batch_cif_ids[0])

    with torch.no_grad():
        atom_bond_fea=Variable(nbr_fea)

    output_local=model_local(atom_bond_fea, atom_num)

    pre_vol=normalizer.denorm(output_local[0].data.cpu())
    pre_target.append(pre_vol.cpu().item())
    
    pre_local_vol=normalizer.denorm(output_local[1].data.cpu())
    local_energy.append(pre_local_vol)#output_local[1].cpu())
    
    atom_tablenumber.append(atom_table_num)
    
#     pre_voltage=torch.cat(pre_target, dim=0)
#     target_voltage=torch.cat(target_vol, dim=0)
#     local_energy_atom=torch.cat(local_energy, dim=0) 
#     atom_tablenum=torch.cat(atom_tablenumber, dim=0)


# In[ ]:


pre_out=[]
cif_ids=[]
atom_tablenum=[]
atom_bond=[]
atom_num=[]
for i in range(len(dataset_CGCNN)):
    pre_out.append(dataset_CGCNN[i][0])  #covalent results
    cif_ids.append(dataset_CGCNN[i][4])
    atom_tablenum.append(dataset_CGCNN[i][3])
    atom_num.append(dataset_CGCNN[i][2])
    atom_bond.append(dataset_CGCNN[i][1])


# In[ ]:


with open ("battery_structure_test1.json", "r") as f :
    battery_structure =  json.loads (f.read ())
    

ids_spacegroup={}
for i in range(len(battery_structure)):
    if battery_structure[i][0]['working_ion']=='Li':
        ids=battery_structure[i][0]['adj_pairs'][0]['structure']['entry_id']
        ids_spacegroup[ids]=battery_structure[i][0]['spacegroup']['number']


# In[ ]:


from cgcnn.number_neighbors import Neighbors
neigbors = Neighbors('../root_dir_std')
neighbors_cif=[]
neighbors=[]
num_neighbors=[]
for nbr in neigbors:
    neighbors_cif.append(nbr[1])
    neighbors.append(nbr[0])
    num=[]
    for nb in nbr[0]:
        #print(nb)
        num_b=len(nb)
        #print(num_b)
        num.append(num_b)
    #print(num)
    num_neighbors.append(num)


# In[ ]:


M_bond_cif={}

for i in range(len(cif_ids)):  #第i个结构
    M_bond=[]
    M_table_num=[]
    local_energy_M=[]
    neighbor_atom_1th=[]
    neighbor_atom_2ed=[]
    M_neighbors=[]
    
    
    p=local_cif_ids.index(cif_ids[i])
    q=neighbors_cif.index(cif_ids[i][0])
    
    for j in range(len(atom_tablenum[i])):  # 第i个结构中的第j个原子

        M_table_num.append(int(atom_tablenum[i][j].item()))
        
        near_1th_num=atom_bond[i][j][0].item()  #第i个结构中的第j个原子 的最近邻原子序号
        neighbor_atom_1th.append(int(atom_tablenum[i][near_1th_num].item()))
        near_2ed_num=atom_bond[i][j][1].item() #第i个结构中的第j个原子 的次近邻原子序号
        neighbor_atom_2ed.append([int(atom_tablenum[i][near_1th_num].item()), int(atom_tablenum[i][near_2ed_num].item())])
        
        M_bond.append(pre_out[i][1][1][j])
        local_energy_M.append(local_energy[p][j].cpu().item())
        
        M_neighbors.append(num_neighbors[q][j])
        
        spacegroup_num=ids_spacegroup[cif_ids[i][0]]
        
        M_bond_cif[cif_ids[i][0]]=[M_bond, M_table_num, neighbor_atom_1th, 
                                   neighbor_atom_2ed, local_energy_M, M_neighbors, spacegroup_num]


# In[ ]:


bond_OMO={}

for keys in M_bond_cif.keys():
    near2=M_bond_cif[keys][3]
    OMO_list=[]
    for i in range(len(near2)):  
        num8=near2[i].count(8)
        
        if num8==2:
            listO=[M_bond_cif[keys][1][i], M_bond_cif[keys][0][i], M_bond_cif[keys][2][i], 
                   M_bond_cif[keys][3][i], M_bond_cif[keys][4][i], M_bond_cif[keys][5][i], M_bond_cif[keys][6]]
            OMO_list.append(listO)
        else:
            continue
            
    if OMO_list==[]:
        continue
            
            
    bond_OMO[keys]=OMO_list


# In[ ]:


bond_OMO_df=pd.DataFrame(columns=range(64))


# In[ ]:


index=0
for keys in bond_OMO.keys():
    for i in range(len(bond_OMO[keys])):
        bond_OMO_df.loc[index]=bond_OMO[keys][i][1].detach().numpy()
        index+=1


# In[ ]:


bond_OMO_df['M_num']=''
bond_OMO_df['local_energy']=''
bond_OMO_df['neighbor_1th']=''
bond_OMO_df['M_neighbor']=''
bond_OMO_df['spacegroup_num']=''
index = 0 
for keys in bond_OMO.keys():
    for i in range(len(bond_OMO[keys])):
        #bond_OMO_df['spacegroup'][index]=M_bond_cif[ids][2]
        bond_OMO_df['M_num'][index]=bond_OMO[keys][i][0]
        bond_OMO_df['local_energy'][index]=bond_OMO[keys][i][4]
        bond_OMO_df['neighbor_1th'][index]=bond_OMO[keys][i][2]
        bond_OMO_df['M_neighbor'][index]=bond_OMO[keys][i][5]
        bond_OMO_df['spacegroup_num'][index]=bond_OMO[keys][i][6]
        index+=1


# In[ ]:


embeddingfeature_PCA_element=pd.read_csv('../data/element_features')


# In[ ]:


num_ele={1:'H', 3:'Li', 5:'B', 6:'C', 7:'N',8:'O', 9:'F', 11:'Na', 
 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 19:'K', 
 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 
 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 
 34:'Se', 35:'Br', 37:'Rb', 38:'Sr', 39:'Y', 40:'Zr', 41:'Nb', 
 42:'Mo', 46:'Pd', 49:'In', 50:'Sn', 51:'Sb', 52:'Te', 55:'Cs', 
 56:'Ba', 57:'La', 73:'Ta', 74:'W', 75:'Re', 83:'Bi'}


# In[ ]:


bond_OMO_df['element']=''
for index in bond_OMO_df.index:
    ele=num_ele[bond_OMO_df['M_num'][index]]
    bond_OMO_df['element'][index]=ele


# In[ ]:


bond_OMO_df['first_ion_E']=''
bond_OMO_df['electronegativity']=''
bond_OMO_df['covalent_radius']=''
bond_OMO_df['atom_radius']=''
#bond_OMO_df=''

for index in bond_OMO_df.index:
    
    element=bond_OMO_df.loc[index]['element']
    bond_OMO_df['first_ion_E'][index]=embeddingfeature_PCA_element['first_ion_E'][element]
    bond_OMO_df['electronegativity'][index]=embeddingfeature_PCA_element['electronegativity'][element]
    bond_OMO_df['covalent_radius'][index]=embeddingfeature_PCA_element['covalent_radius'][element]
    bond_OMO_df['atom_radius'][index]=embeddingfeature_PCA_element['atom_radius'][element]


# In[ ]:


bond_OMO_df['voltage_color']=''
for index in bond_OMO_df.index:
    if bond_OMO_df['local_energy'][index]<=-10:
        bond_OMO_df['voltage_color'][index] =0
        
    elif bond_OMO_df['local_energy'][index]>-10 and bond_OMO_df['local_energy'][index]<=-5:
        bond_OMO_df['voltage_color'][index] =1
    
    elif bond_OMO_df['local_energy'][index] >-5 and bond_OMO_df['local_energy'][index]<=-3:
        bond_OMO_df['voltage_color'][index] =2
    
    elif bond_OMO_df['local_energy'][index] >-3 and bond_OMO_df['local_energy'][index]<=-1:
        bond_OMO_df['voltage_color'][index] =3
    
    elif bond_OMO_df['local_energy'][index] >-1 and bond_OMO_df['local_energy'][index]<=1:
        bond_OMO_df['voltage_color'][index] =4
    
    elif bond_OMO_df['local_energy'][index] >1 and bond_OMO_df['local_energy'][index]<=3:
        bond_OMO_df['voltage_color'][index] =5
    
    elif bond_OMO_df['local_energy'][index] >3 and bond_OMO_df['local_energy'][index]<=5:
        bond_OMO_df['voltage_color'][index] =6
        
    elif bond_OMO_df['local_energy'][index] >5:
        bond_OMO_df['voltage_color'][index] =7


# In[ ]:


bond_OMO_df.to_csv('OMO_local.csv')

