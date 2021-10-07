#!/usr/bin/env python
# coding: utf-8

# In[2]:


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


# In[3]:


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


# In[4]:


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


# In[5]:


class CIFData(object):
    def __init__(self, cif_file, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        atom_init_file='atom_init.json'
        self.cif_file=cif_file
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    def __getitem__(self, cif_file):
        crystal = Structure.from_file(self.cif_file)
        space_group=SpacegroupAnalyzer(crystal).get_space_group_number()
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_file))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(0)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_file, space_group


# In[6]:


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


# In[15]:



def model_path (ion):
    if ion == 'Li':
        modelpath = 'upload_cif_predict/model/model_best.pth.tar'
    elif ion=='Na':
        modelpath = 'upload_cif_predict/model/model_best_Na.pth.tar'
    elif ion=='K':
        modelpath = 'upload_cif_predict/model/model_best_K.pth.tar'
    elif ion=='Rb':
        modelpath = 'upload_cif_predict/model/model_best.pth.tar'
    elif ion=='Cs':
        modelpath = 'upload_cif_predict/model/model_best.pth.tar'
    elif ion=='Mg':
        modelpath = 'upload_cif_predict/model/model_best_Mg.pth.tar'
    elif ion=='Ca':
        modelpath = 'upload_cif_predict/model/model_best_Ca.pth.tar'
    elif ion=='Zn':
        modelpath = 'upload_cif_predict/model/model_best_Zn.pth.tar'
    elif ion=='Y':
        modelpath = 'upload_cif_predict/model/model_best_Y.pth.tar'
    elif ion=='Al':
        modelpath = 'upload_cif_predict/model/model_best_Al.pth.tar'
    else:
        raise ValueError('invalid working_ion')

    return modelpath


# In[16]:


import argparse
from torch.autograd import Variable
import json
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from upload_cif_predict.data import collate_pool
from upload_cif_predict.model import CrystalGraphConvNet

def file_get_result(cif_file, ion):
    data=CIFData(cif_file=cif_file)
    modelpath=model_path (ion)
    model_checkpoint = torch.load(modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    
    structures, _, _, _ = data[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len,
                                n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len,
                                n_h=model_args.n_h,
                                classification=True if model_args.task ==
                                'classification' else False)
    
    normalizer = Normalizer(torch.zeros(3))
    model.load_state_dict(model_checkpoint['state_dict'])
    normalizer.load_state_dict(model_checkpoint['normalizer'])
    
    atom_fea=data[0][0][0]
    nbr_fea=data[0][0][1]
    nbr_fea_idx=data[0][0][2]
    spacegroup=data[0][3]
    atom_index=set()
    for idx in nbr_fea_idx:
        for atoms in idx:
            #print(atoms.item())
            atom_index.add(atoms.item())
    crystal_atom_idx=torch.LongTensor(np.array(list(atom_index)))
    
    input_var = (Variable(atom_fea),
                 Variable(nbr_fea),
                 nbr_fea_idx,
                 [crystal_atom_idx])
    
    model.eval()
    
    output=model(*input_var)
    pred_voltage=normalizer.denorm(output.data.cpu())
    pred_voltage=pred_voltage[0][0].tolist()
    voltage=round(pred_voltage, 2)
    
    
    return cif_file, ion, spacegroup, voltage 

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'cif'}
    UPLOAD_FOLDER = '/file_pred_voltage'
    return '.' in filename and \
       filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
