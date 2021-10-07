#!/usr/bin/env python
# coding: utf-8

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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

bond_OMO_df=pd.read_csv('../data/OMO_local.csv')


bond_OMO_df=bond_OMO_df.drop(['Unnamed: 0'], axis=1)


bond_OMO_df

df_embeddingfeature_PCA_element=pd.read_csv('../data/element_features.csv')

df_embeddingfeature_PCA_element.set_index(['0'], inplace=True)


df_embeddingfeature_PCA_element


num_ele={1:'H', 3:'Li', 5:'B', 6:'C', 7:'N',8:'O', 9:'F', 11:'Na', 
 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 19:'K', 
 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 
 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 
 34:'Se', 35:'Br', 37:'Rb', 38:'Sr', 39:'Y', 40:'Zr', 41:'Nb', 
 42:'Mo', 46:'Pd', 49:'In', 50:'Sn', 51:'Sb', 52:'Te', 55:'Cs', 
 56:'Ba', 57:'La', 73:'Ta', 74:'W', 75:'Re', 83:'Bi'}

ele_dict={}
for ele_num in np.unique(bond_OMO_df['M_num']):
    list_ele=[]
    for index in bond_OMO_df.index:
        if bond_OMO_df['M_num'][index]==ele_num:
            list_ele.append(bond_OMO_df['local_energy'][index])
        else:
            continue
    ele_dict[ele_num]=list_ele


ele_df=pd.DataFrame(columns=['M_num', 'local_voltage', 'element', 'first_ion_E', 'electronegativity', 'covalent_radius', 'atom_radius'])


for index in range(len(ele_dict)):
    ele_df.loc[index]=index
index=0
for keys in ele_dict.keys():
    ele_df['M_num'][index]=keys
    ele_df['local_voltage'][index]=ele_dict[keys]
    #ele_df['ave_voltage'][index]=np.mean(ele_dict[keys])
    #ele_df['mid_voltage'][index]=np.median(ele_dict[keys])
    element=num_ele[keys]
    ele_df['element'][index]=element
    ele_df['first_ion_E'][index]=df_embeddingfeature_PCA_element['first_ion_E'][element]
    ele_df['electronegativity'][index]=df_embeddingfeature_PCA_element['electronegativity'][element]
    ele_df['covalent_radius'][index]=df_embeddingfeature_PCA_element['covalent_radius'][element]
    ele_df['atom_radius'][index]=df_embeddingfeature_PCA_element['atom_radius'][element]
    
    index+=1


for index in ele_df.index:
    exec('box_{}={}'.format(ele_df['element'][index], ele_df['local_voltage'][index]))


boxs=[]
for index in ele_df.index:
    boxs.append(ele_df['local_voltage'][index])
labels=ele_df['element']


plt.figure(figsize=(20,3))
plt.xlim(-0.5, 43.5)
plt.ylim(3.5, 0.2)
#plt.ylabel('Local Voltage (V)',fontsize=30)
plt.xticks(fontsize=16, color="black")
plt.yticks(fontsize=18, color="black")

area = np.pi * 5**2
plt.plot(ele_df['element'], ele_df['atom_radius'], '-o', markersize=10)
#plt.savefig('atom_radius', dpi=100)

import seaborn as sns
plt.figure(figsize=(20,5))
plt.xlim(0.5, 44.5)
plt.ylabel('Local Voltage (V)',fontsize=30)
plt.xticks(fontsize=16, color="black")
plt.yticks(fontsize=18, color="black")

green_diamond = dict(markerfacecolor='black', markersize=5, marker='d')
medianprops = dict(linestyle='-.', linewidth=1.5, color='black')
boxprops = dict(linestyle='-', linewidth=2, color='black', facecolor='purple')


ax=plt.boxplot(boxs, labels=labels, patch_artist = True, boxprops=boxprops, 
               flierprops=green_diamond, whis=2, widths=0.25, medianprops=medianprops)  

#widths:指定箱线图的宽度，默认为0.5 whis=None,    
# 指定上下须与上下四分位的距离，默认为1.5倍的四分位差
#boxprops=None,    # 设置箱体的属性，如边框色，填充色等
#medianprops=None,    # 设置中位数的属性，如线的类型、粗细等；
# area = np.pi * 5**2
# plt.scatter(ele_df['element'], ele_df['atom_radius']*10+15, marker='o', s=area)

#ax.xaxis.grid(True)

plt.savefig('voltage_atom_radius', dip=800)
plt.show()


from matplotlib.pyplot import MultipleLocator

plt.figure(figsize=(20,10))
#plt.xlim(0, 84)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('average voltage',fontsize=20)

miloc = plt.MultipleLocator(1)

ax = plt.gca()
area = np.pi * 5**2
ax.grid(axis='x', which='major', ls='--')
x_major_locator=MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
plt.scatter(ele_df['M_num'], ele_df['covalent_radius'], marker='o')
plt.xticks(ele_df['M_num'], ele_df['element'])
#plt.scatter(ele_df['M_num'], ele_df['ave_voltage'], s=area)
#plt.savefig('ele_voltage_ave')
plt.show()
# plt.legend(loc = 'upper right')


# # OMO local voltage in perodic table


ele_dict={}
for ele_num in np.unique(bond_OMO_df['M_num']):
    list_ele=[]
    for index in bond_OMO_df.index:
        if bond_OMO_df['M_num'][index]==ele_num:
            list_ele.append(bond_OMO_df['local_energy'][index])
        else:
            continue
    ele_dict[ele_num]=list_ele


ele_df=pd.DataFrame(columns=['M_num', 'local_voltage', 'ave_voltage', 'mid_voltage', 'element'])


for index in range(len(ele_dict)):
    ele_df.loc[index]=index
index=0
for keys in ele_dict.keys():
    ele_df['M_num'][index]=keys
    ele_df['local_voltage'][index]=ele_dict[keys]
    ele_df['ave_voltage'][index]=np.mean(ele_dict[keys])
    ele_df['mid_voltage'][index]=np.median(ele_dict[keys])
    
    ele_df['element'][index]=num_ele[keys]
    index+=1


pt_pos={ 'H' : [0, 0],
         'Li': [1, 0], 'Be': [1, 1],  'B': [1, 12],  'C': [1, 13],  'N': [1, 14],  'O': [1, 15],  'F': [1, 16],
         'Na': [2, 0], 'Mg': [2, 1], 'Al': [2, 12], 'Si': [2, 13],  'P': [2, 14],  'S': [2, 15], 'Cl': [2, 16],
          'K': [3, 0], 'Ca': [3, 1], 'Ga': [3, 12], 'Ge': [3, 13], 'As': [3, 14], 'Se': [3, 15], 'Br': [3, 16],
         'Rb': [4, 0], 'Sr': [4, 1], 'In': [4, 12], 'Sn': [4, 13], 'Sb': [4, 14], 'Te': [4, 15],  'I': [4, 16],
         'Cs': [5, 0], 'Ba': [5, 1], 'Tl': [5, 12], 'Pb': [5, 13], 'Bi': [5, 14],

         'Sc': [3, 2], 'Ti': [3, 3],  'V': [3, 4], 'Cr': [3, 5], 'Mn': [3, 6], 'Fe': [3, 7], 'Co': [3, 8], 'Ni': [3, 9], 'Cu': [3, 10], 'Zn': [3, 11], 
          'Y': [4, 2], 'Zr': [4, 3], 'Nb': [4, 4], 'Mo': [4, 5], 'Tc': [4, 6], 'Ru': [4, 7], 'Rh': [4, 8], 'Pd': [4, 9], 'Ag': [4, 10], 'Cd': [4, 11], 
                       'Hf': [5, 3], 'Ta': [5, 4],  'W': [5, 5], 'Re': [5, 6], 'Os': [5, 7], 'Ir': [5, 8], 'Pt': [5, 9], 'Au': [5, 10], 'Hg': [5, 11],
        
         'La': [6, 2], 'Ce': [6, 3], 'Tb': [6, 10], 'Dy': [6, 11], 'Ho': [6, 12], 'Er': [6, 13], 'Tm': [6, 14], 'Lu': [6, 16]
}



pt = [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
      [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
      [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
      [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
      [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
      [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
      [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0]]


for keys in ele_dict.keys():
    i = pt_pos[num_ele[keys]][0]
    j = pt_pos[num_ele[keys]][1]
    pt[i][j]=np.median(ele_dict[keys])



from PIL import Image
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.pyplot as plt



pt_np = np.array (pt)


cmap = plt.cm.get_cmap('seismic')
norm = mpl.colors.Normalize(vmin=-1, vmax=6.5)
bounds = [ round(elem, 2) for elem in np.linspace(0, 6, 59)] 

from matplotlib.colors import LinearSegmentedColormap
interval = np.hstack([np.linspace(0, 0.4), np.linspace(0.6, 1)])
colors = plt.cm.RdBu_r(interval)
cmap = LinearSegmentedColormap.from_list('name', colors)
norm = mpl.colors.Normalize(vmin=-1.5, vmax=7.5)

plt.figure()
plt.figure(figsize=(15,10))
ax = plt.gca()

Table=plt.imshow(pt_np, cmap=cmap)
my_x_ticks = np.arange(-0.5, 16.5, 1) # 坐标范围，颜色，字体
my_y_ticks = np.arange(-0.5, 7.5, 1) # 坐标范围，颜色，字体
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
# plt.xticks([])
# plt.yticks([])
# ax.set_xticks([])  
# ax.set_yticks([]) 
#plt.xticks(fontsize=25, color="black", rotation=45) # 坐标字体大小，颜色
#plt.yticks(fontsize=25, color="black", rotation=0) #  坐标字体大小，颜色
#plt.minorticks_on()
plt.tick_params(width=0)  #坐标刻度消失
ax.spines['top'].set_color('None') #边框线消失
ax.spines['right'].set_color('black') #边框线消失
ax.spines['left'].set_color('black') #边框线消失
ax.spines['bottom'].set_color('black')  #边框线消失
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax) # colorbar 位置，距离主图位置，宽度
cax = divider.append_axes("right", size="3%", pad=0.1) # colorbar 位置，距离主图位置，宽度
cb = plt.colorbar(Table, cax=cax)
cb.set_label('Voltage (V)') # colorbar 标注
plt.rcParams['font.size'] = 24 # colorbar 字体大小
ax.grid(color = 'w',linestyle='-',linewidth = 1) # 网格线
#plt.savefig('Table.png', dpi=1200)


ele_count={}
for ele_num in np.unique(bond_OMO_df['element']):
    total_size=len(bond_OMO_df)
    ele_num_count=list(bond_OMO_df['element']).count(ele_num)
    ele_num_ratio=ele_num_count/total_size
    ele_count[ele_num]=ele_num_count


bond_OMO_df['local_energy'].hist(bins=50,alpha = 0.5)


# # Corelation between PCA1 PCA2 and element features

bond_OMO_df4=bond_OMO_df.drop(['voltage_color', 'neighbor_1th', 'M_num', 'local_energy', 
                               'element', 'M_neighbor', 'spacegroup_num', 'first_ion_E', 
                               'electronegativity', 'covalent_radius', 'atom_radius'], axis=1)
# bond_OMO_df1=bond_OMO_df0.drop(['neighbor_1th'], axis=1)
# bond_OMO_df2=bond_OMO_df1.drop(['M_num'], axis=1)
# bond_OMO_df3=bond_OMO_df2.drop(['local_energy'], axis=1)
# bond_OMO_df4=bond_OMO_df3.drop(['element'], axis=1)

pca_breast = PCA(n_components=2)

M_bond_PCA = pca_breast.fit_transform(bond_OMO_df4)


M_bond_PCA_df = pd.DataFrame(data = M_bond_PCA, columns = ['PCA1', 'PCA2'])


M_bond_PCA_df['M_num']= bond_OMO_df['M_num']
#M_bond_PCA_df['element']= bond_OMO_df['element']
M_bond_PCA_df['local_energy']= bond_OMO_df['local_energy']
M_bond_PCA_df['voltage_color']=bond_OMO_df['voltage_color']
M_bond_PCA_df['M_neighbor']=bond_OMO_df['M_neighbor']
M_bond_PCA_df['first_ion_E']=bond_OMO_df['first_ion_E']
M_bond_PCA_df['electronegativity']=bond_OMO_df['electronegativity']
M_bond_PCA_df['covalent_radius']=bond_OMO_df['covalent_radius']
M_bond_PCA_df['atom_radius']=bond_OMO_df['atom_radius']
M_bond_PCA_df['spacegroup']=bond_OMO_df['spacegroup_num']


M_bond_PCA_df


traindata=pd.DataFrame(M_bond_PCA_df,dtype=np.float)

M_bond_PCA_df_type=traindata.corr()


plt.figure()
plt.figure(figsize=(13,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)

ax=sns.heatmap(M_bond_PCA_df_type, center=0, fmt='d')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#plt.savefig('local_vol_corr', bbox_inches = 'tight')
plt.show()

cm=plt.cm.get_cmap('RdYlBu_r')

plt.figure()
plt.figure(figsize=(19,19))
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel('Dimension1',fontsize=28)
plt.ylabel('Dimension2',fontsize=28)
plt.tick_params(axis='both',direction='in', which='major',width=4, length=10)
#plt.title("Principal Component Analysis",fontsize=20)
# plt.xlim(-5.5, 15)
# plt.ylim(-7.5, 11)

colors =M_bond_PCA_df['voltage_color']
num_classes = len(np.unique(colors))+1
palette = np.array(sns.color_palette("hls", num_classes))
c=palette[colors.astype(np.int)]
cm=plt.cm.get_cmap('RdYlBu_r')
area=np.array(colors)
area = np.pi * 3**2

plt.scatter(M_bond_PCA_df['PCA1'], M_bond_PCA_df['PCA2'], s=area, c=c, cmap=cm)
# plt.legend(loc = 'upper right')
# for indexs, row in M_bond_PCA_df3.iterrows():
#     plt.annotate(M_bond_PCA_df3.loc[indexs]['element'], xy = (M_bond_PCA_df3.loc[indexs]['PCA1'], M_bond_PCA_df3.loc[indexs]['PCA2']), 
#                  xytext = (M_bond_PCA_df3.loc[indexs]['PCA1'], M_bond_PCA_df3.loc[indexs]['PCA2']), fontsize=8) 
 #   print (index)

plt.colorbar()
#plt.savefig('PCA_bondO')

plt.show()


# # t-SNE


M_bond_PCA_df3=pd.DataFrame(columns=M_bond_PCA_df.columns)
from random import sample
pick_sample=sample(range(len(M_bond_PCA_df)), 300)
index=0
for i in pick_sample:
    M_bond_PCA_df3.loc[index]=M_bond_PCA_df.iloc[i]
    index+=1

time_start = time.time()
RS = 123

fashion_tsne = TSNE(random_state=RS).fit_transform(bond_OMO_df4)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


bond_OMO_df.index = range(len(bond_OMO_df))

M_bond_tsne_df = pd.DataFrame(data = fashion_tsne, columns = ['tsen1', 'tsen2'])
M_bond_tsne_df['local_energy']= bond_OMO_df['local_energy']
M_bond_tsne_df['table_num']= bond_OMO_df['M_num']
M_bond_tsne_df['element']= bond_OMO_df['element']
M_bond_tsne_df['voltage_color']= bond_OMO_df['voltage_color'] 
M_bond_tsne_df['M_neighbor']= bond_OMO_df['M_neighbor'] 
M_bond_tsne_df['spacegroup']= bond_OMO_df['spacegroup_num'] 


M_bond_tsne_df.to_csv('OMO_tsne.csv')

M_bond_tsne_df['local_energy'].hist(bins=50,alpha = 0.5)


def fashion_scatter(x, y, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))+1
    palette = np.array(sns.color_palette("hls", num_classes))
    c=palette[colors.astype(np.int)]

    # create a scatter plot.
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    area = np.pi * 3**2
    cm=plt.cm.get_cmap('RdYlBu_r')
    
    #c=colors
    sc = ax.scatter(x, y, s=area, c=c, cmap=cm)
#     plt.xlim(-25, 25)
#     plt.ylim(-25, 25)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    ax.axis('off')
    ax.axis('tight')
    #plt.savefig('C:\\Users\\zhang\\workshop-2017\\cgcnn-master\\my_cgcnn\\Li_battery_voltage\\predict_analysis\\t-SNE_O-M-O_neighbors')
    #sc.colorbar()
    plt.show()


fashion_scatter(M_bond_tsne_df['tsen1'], M_bond_tsne_df['tsen2'], M_bond_tsne_df['voltage_color'])


colors =M_bond_tsne_df['voltage_color']
num_classes = len(np.unique(colors))+1
#palette = np.array(sns.color_palette("hls", num_classes))
palette=sns.color_palette("hls", num_classes)
sns.palplot(palette)
#plt.savefig('C:\\Users\\zhang\\workshop-2017\\cgcnn-master\\my_cgcnn\\Li_battery_voltage\\predict_analysis\\t-SNE_local_energy_colorbar')

fashion_scatter(M_bond_tsne_df['tsen1'], M_bond_tsne_df['tsen2'], M_bond_tsne_df['M_neighbor'])


N_neighbors_df = pd.DataFrame(columns = M_bond_tsne_df.columns)
for index in M_bond_tsne_df.index:
    if M_bond_tsne_df['M_neighbor'][index]>15:
        N_neighbors_df.loc[index]=M_bond_tsne_df.loc[index]


plt.figure()
plt.figure(figsize=(10,10))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel('tsen1',fontsize=20)
# plt.ylabel('tsen2',fontsize=20)
# plt.title("Principal Component Analysis",fontsize=20)
#targets = list(embedding_features.index)

M_bond_tsen_pices=M_bond_tsne_df.loc[3000:5000]

colors =M_bond_tsen_pices['voltage_color']
num_classes = len(np.unique(colors))+1
palette = np.array(sns.color_palette("hls", num_classes))
c=palette[colors.astype(np.int)]


cm=plt.cm.get_cmap('RdYlBu_r')
#area=np.array(colors)
area = np.pi * 3**2

plt.scatter(M_bond_tsen_pices['tsen1'], M_bond_tsen_pices['tsen2'], s=area, c=c, cmap=cm)
# plt.legend(loc = 'upper right')
for indexs, row in M_bond_tsen_pices.iterrows():
    plt.annotate(M_bond_tsen_pices.loc[indexs]['element'], xy = (M_bond_tsen_pices.loc[indexs]['tsen1'], M_bond_tsen_pices.loc[indexs]['tsen2']), 
                 xytext = (M_bond_tsen_pices.loc[indexs]['tsen1'], M_bond_tsen_pices.loc[indexs]['tsen2']), fontsize=8) 
 #   print (index)

#plt.colorbar()
# ax.axis('off')
# ax.axis('tight')
#plt.savefig('C:\\Users\\zhang\\workshop-2017\\cgcnn-master\\my_cgcnn\\Li_battery_voltage\\predict_analysis\\t-SNE_O-M-O_local_energy_2000')

plt.show()





