#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
sys.path.append('../')
from cgcnn.data_pre import CIFData
from cgcnn.data_pre import collate_pool
from cgcnn.model_pre import CrystalGraphConvNet

modelpath='../trained_model/model_best.pth.tar'  # we need a model to product features in every layers

#dataset = CIFData('../../Li_battery_voltage/root_dir_2std_2190')
dataset = CIFData('../data/std15')
collate_fn = collate_pool
test_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=0, collate_fn=collate_fn,
                        pin_memory=False)

structures, _, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]
model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                            atom_fea_len=64,
                            n_conv=3,
                            h_fea_len=128,
                            n_h=1,
                            classification=False)

class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

criterion = nn.MSELoss()
normalizer = Normalizer(torch.zeros(3))
checkpoint = torch.load(modelpath, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
normalizer.load_state_dict(checkpoint['normalizer'])


# # get all the features in every layers in the model

atom_tablenum=[]
pre_out=[]
cif_ids=[]
target_var = []
wrong_ids = []
for i, (input, target, batch_cif_ids, atom_table_num) in enumerate(test_loader):
    with torch.no_grad():
        input_var = (Variable(input[0]),
                    Variable(input[1]),
                    input[2],
                    input[3])

    try:
        pre_out.append(model(*input_var))
        atom_tablenum.append(atom_table_num)
        cif_ids.append(batch_cif_ids)
        target_var.append(target)
    except ValueError:
        wrong_ids.append(batch_cif_ids)

    '''
    pre_out[i][0] the predicted voltage values.
    pre_out[i][1] the output of every layer
    pre_out[i][1][0] embedding
    pre_out[i][1][1] conv
    pre_out[i][1][2] pooling
    pre_out[i][1][3] hidden layer
    '''
# now, we get all the features in every layers


# # put the features in the embedding layer into a dict


embedding_layer_features={}
for i in range(len(pre_out)):
    for j in range(len(pre_out[i][1][0])):
        keys=int(atom_tablenum[i][j].item())
        values=pre_out[i][1][0][j].detach().numpy()
        embedding_layer_features[keys]=values
# put the features from the embedding layer into  dict--embedding_layer_features
# Then, we tried to save the dict to .json, but filed


# # change the atom features into a dataframe


import pandas as pd
embedding_features=pd.DataFrame.from_dict(embedding_layer_features, orient='index')


num_ele={1:'H', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N',8:'O', 9:'F', 11:'Na', 
 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 19:'K', 
 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 
 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 
 34:'Se', 35:'Br', 37:'Rb', 38:'Sr', 39:'Y', 40:'Zr', 41:'Nb', 
 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd', 47: 'Ag', 48:'Cd', 
 49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I', 55:'Cs', 56:'Ba', 
 57:'La', 58:'Ce', 59:'Pr', 60:'Nd', 72:'Hf', 73:'Ta', 74:'W', 
 75:'Re', 76:'Os', 77:'Ir', 78:'Pt', 79:'Au', 80:'Hg', 81: 'Tl', 83:'Bi'}


embedding_features=embedding_features.rename(index=num_ele) 
# change the atom table number into correspodding atom name


# # Principle Composition Analysis

from sklearn.decomposition import PCA

pca_breast = PCA(n_components=2) # n_components=int how many conpoenet left. n_compoenent=float 留下的主成分占总成份的多少

# mle 自己选择降维维度

embeddingfeatures_PCA = pca_breast.fit(embedding_features)


embeddingfeatures_PCA = pca_breast.fit_transform(embedding_features)


embeddingfeature_PCA_df = pd.DataFrame(data = embeddingfeatures_PCA, columns = ['PCA1', 'PCA2'])
embeddingfeature_PCA_cor=embeddingfeature_PCA_df.corr()

embeddingfeature_PCA_df['index']= embedding_features.index


embeddingfeature_PCA_df.set_index(['index'], inplace=True)


# # add atom features into

embeddingfeature_PCA_df['atom_radius']=''
embeddingfeature_PCA_df['covalent_radius']=''
embeddingfeature_PCA_df['electronegativity']=''
embeddingfeature_PCA_df['first_ion_E']=''  #eV

data_element = pd.read_csv('../data/element_features.csv', index_col=0)

for index, row in embeddingfeature_PCA_df.iterrows():
    embeddingfeature_PCA_df['atom_radius'][index]=data_element['atom_radius'][index]
    embeddingfeature_PCA_df['covalent_radius'][index]=data_element['covalent_radius'][index]
    embeddingfeature_PCA_df['electronegativity'][index]=data_element['electronegativity'][index]
    embeddingfeature_PCA_df['first_ion_E'][index]=data_element['first_ion_E'][index]

import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(12,12))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Dim2',fontsize=38)
plt.ylabel('Covalent Radius',fontsize=38)
plt.plot(embeddingfeature_PCA_df['PCA2'], embeddingfeature_PCA_df['covalent_radius'], 'o', markersize=30)
# for index, row in embeddingfeature_PCA_df.iterrows():
#     plt.annotate(index, xy = (embeddingfeature_PCA_df.loc[index]['PCA2'], embeddingfeature_PCA_df.loc[index]['covalent_radius']), 
#                  xytext = (embeddingfeature_PCA_df.loc[index]['PCA2'], embeddingfeature_PCA_df.loc[index]['covalent_radius']), 
#                  fontsize=45, fontproperties='Arial') 
plt.savefig('PCA2_covalentradius')



embedding_feature=pd.DataFrame(embeddingfeature_PCA_df,dtype=np.float)
embedding_feature_cor=embedding_feature.corr(method='spearman')


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,8))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax=sns.heatmap(embedding_feature_cor, center=0, fmt='d') 
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


Alkali=['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
Alkaline_earth=['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
TM_I=['Sc', 'Y', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re']
TM_II=['Fe', 'Co', 'Ni', 'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']
Post_transition=[ 'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi', 'Po']
Metalliod=['B', 'N', 'O', 'C', 'Si', 'Ge', 'P', 'As', 'Sb', 'S', 'Se', 'Te']
Halogen=['F', 'Cl', 'Br', 'I', 'At']

Alkali_df=pd.DataFrame(columns=embeddingfeature_PCA_df.columns)
Alkaline_earth_df=pd.DataFrame(columns=embeddingfeature_PCA_df.columns)
TM_I_df=pd.DataFrame(columns=embeddingfeature_PCA_df.columns)
TM_II_df=pd.DataFrame(columns=embeddingfeature_PCA_df.columns)
Post_transition_df=pd.DataFrame(columns=embeddingfeature_PCA_df.columns)
Metalliod_df=pd.DataFrame(columns=embeddingfeature_PCA_df.columns)
Halogen_df=pd.DataFrame(columns=embeddingfeature_PCA_df.columns)

for index, row in embeddingfeature_PCA_df.iterrows():
    if index in Alkali:
        Alkali_df.loc[index]=embeddingfeature_PCA_df.loc[index]
    elif index in Alkaline_earth:
        Alkaline_earth_df.loc[index]=embeddingfeature_PCA_df.loc[index]
    elif index in TM_I:
        TM_I_df.loc[index]=embeddingfeature_PCA_df.loc[index]
    elif index in TM_II:
        TM_II_df.loc[index]=embeddingfeature_PCA_df.loc[index]
    elif index in Post_transition:
        Post_transition_df.loc[index]=embeddingfeature_PCA_df.loc[index]
    elif index in Metalliod:
        Metalliod_df.loc[index]=embeddingfeature_PCA_df.loc[index]
    elif index in Halogen:
        Halogen_df.loc[index]=embeddingfeature_PCA_df.loc[index]
    else:
        print(index)


import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(18,18))
plt.xticks(fontsize=38)
plt.yticks(fontsize=38)
plt.xlabel('Dim1',fontsize=38)
plt.ylabel('Dim2',fontsize=38)
#plt.tick_params(axis='both',direction='none', which='major',width=4, length=10)
#plt.title("Principal Component Analysis",fontsize=22)

#targets = list(embedding_features.index)
area = np.pi * 10**2


plt_1=plt.scatter(Alkali_df['PCA1'], Alkali_df['PCA2'], marker='o', s=area, c='deeppink', label='Alkali')
plt_2=plt.scatter(Alkaline_earth_df['PCA1'], Alkaline_earth_df['PCA2'], marker='^', s=area, c='orchid', label='Alkaline_Earth')
plt_3=plt.scatter(TM_I_df['PCA1'], TM_I_df['PCA2'], marker='D', s=area, c='slateblue', label='Transition_Metal_I')
plt_3=plt.scatter(TM_II_df['PCA1'], TM_II_df['PCA2'], marker='d', s=area, c='indigo', label='Transition_Metal_II')
plt_4=plt.scatter(Post_transition_df['PCA1'], Post_transition_df['PCA2'], marker='p', s=area, c='yellowgreen', label='Post_Transition_Metal')
plt_5=plt.scatter(Metalliod_df['PCA1'], Metalliod_df['PCA2'], marker='s', s=area, c='deepskyblue', label='Metalliod')
plt_6=plt.scatter(Halogen_df['PCA1'], Halogen_df['PCA2'], marker='*', s=area, c='y', label='Halogen')

legend_front={"family" : "Arial"}
plt.legend(loc = 'upper left',  prop={'size':40, "family" : "Arial"}, facecolor='white')

for index, row in embeddingfeature_PCA_df.iterrows():
    plt.annotate(index, xy = (embeddingfeature_PCA_df.loc[index]['PCA1'], embeddingfeature_PCA_df.loc[index]['PCA2']), 
                 xytext = (embeddingfeature_PCA_df.loc[index]['PCA1'], embeddingfeature_PCA_df.loc[index]['PCA2']), fontsize=40, fontproperties='Arial') 
plt.savefig('PCA_element_group', bbox_inches = 'tight')
plt.show()

