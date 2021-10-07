#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


embeddingfeature_PCA_element=pd.DataFrame(columns=['first_ion_E', 'electronegativity', 'covalent_radius', 'atom_radius'])


# In[3]:


embeddingfeature_PCA_element={}


# In[4]:


embeddingfeature_PCA_element['first_ion_E']={}
embeddingfeature_PCA_element['first_ion_E']['H']=13.595
embeddingfeature_PCA_element['first_ion_E']['Li']=5.39
embeddingfeature_PCA_element['first_ion_E']['B']=8.296
embeddingfeature_PCA_element['first_ion_E']['C']=11.256
embeddingfeature_PCA_element['first_ion_E']['N']=14.53
embeddingfeature_PCA_element['first_ion_E']['O']=13.614
embeddingfeature_PCA_element['first_ion_E']['F']=17.418
embeddingfeature_PCA_element['first_ion_E']['Na']=5.138
embeddingfeature_PCA_element['first_ion_E']['Mg']=7.644
embeddingfeature_PCA_element['first_ion_E']['Al']=5.894
embeddingfeature_PCA_element['first_ion_E']['Si']=8.149
embeddingfeature_PCA_element['first_ion_E']['P']=10.484
embeddingfeature_PCA_element['first_ion_E']['S']=10.357
embeddingfeature_PCA_element['first_ion_E']['Cl']=13.01
embeddingfeature_PCA_element['first_ion_E']['K']=4.339
embeddingfeature_PCA_element['first_ion_E']['Ca']=6.111
embeddingfeature_PCA_element['first_ion_E']['Sc']=6.54
embeddingfeature_PCA_element['first_ion_E']['Ti']=6.82
embeddingfeature_PCA_element['first_ion_E']['V']=6.74
embeddingfeature_PCA_element['first_ion_E']['Cr']=6.764
embeddingfeature_PCA_element['first_ion_E']['Mn']=7.432
embeddingfeature_PCA_element['first_ion_E']['Fe']=7.87
embeddingfeature_PCA_element['first_ion_E']['Co']=7.86
embeddingfeature_PCA_element['first_ion_E']['Ni']=7.633
embeddingfeature_PCA_element['first_ion_E']['Cu']=7.724
embeddingfeature_PCA_element['first_ion_E']['Zn']=9.391
embeddingfeature_PCA_element['first_ion_E']['Ga']=6
embeddingfeature_PCA_element['first_ion_E']['Ge']=7.88
embeddingfeature_PCA_element['first_ion_E']['As']=9.81
embeddingfeature_PCA_element['first_ion_E']['Se']=9.75
embeddingfeature_PCA_element['first_ion_E']['Br']=11.84
embeddingfeature_PCA_element['first_ion_E']['Rb']=11.84
embeddingfeature_PCA_element['first_ion_E']['Sr']=5.692
embeddingfeature_PCA_element['first_ion_E']['Y']=6.38
embeddingfeature_PCA_element['first_ion_E']['Zr']=6.84
embeddingfeature_PCA_element['first_ion_E']['Nb']=6.88
embeddingfeature_PCA_element['first_ion_E']['Mo']=7.10
embeddingfeature_PCA_element['first_ion_E']['Pd']=8.33
embeddingfeature_PCA_element['first_ion_E']['In']=5.785
embeddingfeature_PCA_element['first_ion_E']['Sn']=7.342
embeddingfeature_PCA_element['first_ion_E']['Sb']=8.639
embeddingfeature_PCA_element['first_ion_E']['Te']=9.01
embeddingfeature_PCA_element['first_ion_E']['Cs']=3.893
embeddingfeature_PCA_element['first_ion_E']['Ba']=5.21
embeddingfeature_PCA_element['first_ion_E']['La']=5.61
embeddingfeature_PCA_element['first_ion_E']['Ta']=7.88
embeddingfeature_PCA_element['first_ion_E']['W']=7.98
embeddingfeature_PCA_element['first_ion_E']['Re']=7.87
embeddingfeature_PCA_element['first_ion_E']['Bi']=7.287


# In[5]:


embeddingfeature_PCA_element['electronegativity']={}

embeddingfeature_PCA_element['electronegativity']['H']=2.18
embeddingfeature_PCA_element['electronegativity']['Li']=0.98
embeddingfeature_PCA_element['electronegativity']['B']=2.04
embeddingfeature_PCA_element['electronegativity']['C']=2.55
embeddingfeature_PCA_element['electronegativity']['N']=3.04
embeddingfeature_PCA_element['electronegativity']['O']=3.44
embeddingfeature_PCA_element['electronegativity']['F']=3.98
embeddingfeature_PCA_element['electronegativity']['Na']=0.98
embeddingfeature_PCA_element['electronegativity']['Mg']=1.31
embeddingfeature_PCA_element['electronegativity']['Al']=1.61
embeddingfeature_PCA_element['electronegativity']['Si']=1.90
embeddingfeature_PCA_element['electronegativity']['P']=2.19
embeddingfeature_PCA_element['electronegativity']['S']=2.58
embeddingfeature_PCA_element['electronegativity']['Cl']=3.16
embeddingfeature_PCA_element['electronegativity']['K']=0.82
embeddingfeature_PCA_element['electronegativity']['Ca']=1.00
embeddingfeature_PCA_element['electronegativity']['Sc']=1.36
embeddingfeature_PCA_element['electronegativity']['Ti']=1.54
embeddingfeature_PCA_element['electronegativity']['V']=1.63
embeddingfeature_PCA_element['electronegativity']['Cr']=1.66
embeddingfeature_PCA_element['electronegativity']['Mn']=1.55
embeddingfeature_PCA_element['electronegativity']['Fe']=1.80
embeddingfeature_PCA_element['electronegativity']['Co']=1.88
embeddingfeature_PCA_element['electronegativity']['Ni']=1.91
embeddingfeature_PCA_element['electronegativity']['Cu']=1.90
embeddingfeature_PCA_element['electronegativity']['Zn']=1.65
embeddingfeature_PCA_element['electronegativity']['Ga']=1.81
embeddingfeature_PCA_element['electronegativity']['Ge']=2.01
embeddingfeature_PCA_element['electronegativity']['As']=2.18
embeddingfeature_PCA_element['electronegativity']['Se']=2.55
embeddingfeature_PCA_element['electronegativity']['Br']=2.96
embeddingfeature_PCA_element['electronegativity']['Rb']=0.82
embeddingfeature_PCA_element['electronegativity']['Sr']=0.95
embeddingfeature_PCA_element['electronegativity']['Y']=1.22
embeddingfeature_PCA_element['electronegativity']['Zr']=1.33
embeddingfeature_PCA_element['electronegativity']['Nb']=1.60
embeddingfeature_PCA_element['electronegativity']['Mo']=2.16
embeddingfeature_PCA_element['electronegativity']['Pd']=2.20
embeddingfeature_PCA_element['electronegativity']['In']=1.78
embeddingfeature_PCA_element['electronegativity']['Sn']=1.96
embeddingfeature_PCA_element['electronegativity']['Sb']=2.05
embeddingfeature_PCA_element['electronegativity']['Te']=2.10
embeddingfeature_PCA_element['electronegativity']['Cs']=0.79
embeddingfeature_PCA_element['electronegativity']['Ba']=0.89
embeddingfeature_PCA_element['electronegativity']['La']=1.10
embeddingfeature_PCA_element['electronegativity']['Ta']=1.50
embeddingfeature_PCA_element['electronegativity']['W']=2.36
embeddingfeature_PCA_element['electronegativity']['Re']=1.90
embeddingfeature_PCA_element['electronegativity']['Bi']=2.02


# In[6]:


embeddingfeature_PCA_element['covalent_radius']={}
embeddingfeature_PCA_element['covalent_radius']['H']=0.32
embeddingfeature_PCA_element['covalent_radius']['Li']=1.33
embeddingfeature_PCA_element['covalent_radius']['B']=0.85
embeddingfeature_PCA_element['covalent_radius']['C']=0.75
embeddingfeature_PCA_element['covalent_radius']['N']=0.71
embeddingfeature_PCA_element['covalent_radius']['O']=0.63
embeddingfeature_PCA_element['covalent_radius']['F']=0.64
embeddingfeature_PCA_element['covalent_radius']['Na']=1.55
embeddingfeature_PCA_element['covalent_radius']['Mg']=1.39
embeddingfeature_PCA_element['covalent_radius']['Al']=1.26
embeddingfeature_PCA_element['covalent_radius']['Si']=1.16
embeddingfeature_PCA_element['covalent_radius']['P']=1.11
embeddingfeature_PCA_element['covalent_radius']['S']=1.03
embeddingfeature_PCA_element['covalent_radius']['Cl']=0.99
embeddingfeature_PCA_element['covalent_radius']['K']=1.96
embeddingfeature_PCA_element['covalent_radius']['Ca']=1.71
embeddingfeature_PCA_element['covalent_radius']['Sc']=1.48
embeddingfeature_PCA_element['covalent_radius']['Ti']=1.36
embeddingfeature_PCA_element['covalent_radius']['V']=1.34
embeddingfeature_PCA_element['covalent_radius']['Cr']=1.22
embeddingfeature_PCA_element['covalent_radius']['Mn']=1.19
embeddingfeature_PCA_element['covalent_radius']['Fe']=1.16
embeddingfeature_PCA_element['covalent_radius']['Co']=1.11
embeddingfeature_PCA_element['covalent_radius']['Ni']=1.10
embeddingfeature_PCA_element['covalent_radius']['Cu']=1.12
embeddingfeature_PCA_element['covalent_radius']['Zn']=1.18
embeddingfeature_PCA_element['covalent_radius']['Ga']=1.24
embeddingfeature_PCA_element['covalent_radius']['Ge']=1.24
embeddingfeature_PCA_element['covalent_radius']['As']=1.21
embeddingfeature_PCA_element['covalent_radius']['Se']=1.16
embeddingfeature_PCA_element['covalent_radius']['Br']=1.14
embeddingfeature_PCA_element['covalent_radius']['Rb']=2.10
embeddingfeature_PCA_element['covalent_radius']['Sr']=1.85
embeddingfeature_PCA_element['covalent_radius']['Y']=1.63
embeddingfeature_PCA_element['covalent_radius']['Zr']=1.54
embeddingfeature_PCA_element['covalent_radius']['Nb']=1.47
embeddingfeature_PCA_element['covalent_radius']['Mo']=1.38
embeddingfeature_PCA_element['covalent_radius']['Pd']=1.20
embeddingfeature_PCA_element['covalent_radius']['In']=1.42
embeddingfeature_PCA_element['covalent_radius']['Sn']=1.40
embeddingfeature_PCA_element['covalent_radius']['Sb']=1.40
embeddingfeature_PCA_element['covalent_radius']['Te']=1.36
embeddingfeature_PCA_element['covalent_radius']['Cs']=2.32
embeddingfeature_PCA_element['covalent_radius']['Ba']=1.96
embeddingfeature_PCA_element['covalent_radius']['La']=1.80
embeddingfeature_PCA_element['covalent_radius']['Ta']=1.46
embeddingfeature_PCA_element['covalent_radius']['W']=1.37
embeddingfeature_PCA_element['covalent_radius']['Re']=1.31
embeddingfeature_PCA_element['covalent_radius']['Bi']=1.51


# In[7]:


embeddingfeature_PCA_element['atom_radius']={}
embeddingfeature_PCA_element['atom_radius']['H']=0.53
embeddingfeature_PCA_element['atom_radius']['Li']=1.67
embeddingfeature_PCA_element['atom_radius']['B']=0.87
embeddingfeature_PCA_element['atom_radius']['C']=0.67
embeddingfeature_PCA_element['atom_radius']['N']=0.56
embeddingfeature_PCA_element['atom_radius']['O']=0.48
embeddingfeature_PCA_element['atom_radius']['F']=0.42
embeddingfeature_PCA_element['atom_radius']['Na']=1.90
embeddingfeature_PCA_element['atom_radius']['Mg']=1.45
embeddingfeature_PCA_element['atom_radius']['Al']=1.18
embeddingfeature_PCA_element['atom_radius']['Si']=1.11
embeddingfeature_PCA_element['atom_radius']['P']=0.98
embeddingfeature_PCA_element['atom_radius']['S']=0.88
embeddingfeature_PCA_element['atom_radius']['Cl']=0.79
embeddingfeature_PCA_element['atom_radius']['K']=2.43
embeddingfeature_PCA_element['atom_radius']['Ca']=1.94
embeddingfeature_PCA_element['atom_radius']['Sc']=1.84
embeddingfeature_PCA_element['atom_radius']['Ti']=1.76
embeddingfeature_PCA_element['atom_radius']['V']=1.71
embeddingfeature_PCA_element['atom_radius']['Cr']=1.66
embeddingfeature_PCA_element['atom_radius']['Mn']=1.61
embeddingfeature_PCA_element['atom_radius']['Fe']=1.56
embeddingfeature_PCA_element['atom_radius']['Co']=1.52
embeddingfeature_PCA_element['atom_radius']['Ni']=1.49
embeddingfeature_PCA_element['atom_radius']['Cu']=1.45
embeddingfeature_PCA_element['atom_radius']['Zn']=1.42
embeddingfeature_PCA_element['atom_radius']['Ga']=1.36
embeddingfeature_PCA_element['atom_radius']['Ge']=1.25
embeddingfeature_PCA_element['atom_radius']['As']=1.14
embeddingfeature_PCA_element['atom_radius']['Se']=1.03
embeddingfeature_PCA_element['atom_radius']['Br']=0.94
embeddingfeature_PCA_element['atom_radius']['Rb']=2.65
embeddingfeature_PCA_element['atom_radius']['Sr']=2.19
embeddingfeature_PCA_element['atom_radius']['Y']=2.12
embeddingfeature_PCA_element['atom_radius']['Zr']=2.06
embeddingfeature_PCA_element['atom_radius']['Nb']=1.98
embeddingfeature_PCA_element['atom_radius']['Mo']=1.90
embeddingfeature_PCA_element['atom_radius']['Pd']=1.69
embeddingfeature_PCA_element['atom_radius']['In']=1.56
embeddingfeature_PCA_element['atom_radius']['Sn']=1.45
embeddingfeature_PCA_element['atom_radius']['Sb']=1.33
embeddingfeature_PCA_element['atom_radius']['Te']=1.23
embeddingfeature_PCA_element['atom_radius']['Cs']=2.98
embeddingfeature_PCA_element['atom_radius']['Ba']=2.53
embeddingfeature_PCA_element['atom_radius']['La']=1.87
embeddingfeature_PCA_element['atom_radius']['Ta']=2.00
embeddingfeature_PCA_element['atom_radius']['W']=1.93
embeddingfeature_PCA_element['atom_radius']['Re']=1.88
embeddingfeature_PCA_element['atom_radius']['Bi']=1.43


# In[8]:


df_embeddingfeature_PCA_element=pd.DataFrame.from_dict(embeddingfeature_PCA_element)


# In[9]:


df_embeddingfeature_PCA_element.to_csv('element_features.csv', index=False)


# In[ ]:




