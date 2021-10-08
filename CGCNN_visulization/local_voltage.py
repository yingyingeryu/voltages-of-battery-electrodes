import torch
import torch.nn as nn
class LocalEnergy(nn.Module):
    
    def __init__ (self, atom_bond_fea_len=64):
        super(LocalEnergy, self).__init__()
        self.local_energy=nn.Linear(atom_bond_fea_len, 1)
        #self.hidden_energy=nn.Linear(32, 1)
        #self.local_energy_softplus = nn.Softplus()
        
    #model = nn.Linear(64, 1)
    def forward(self, atom_bond_fea, crystal_atom_idx):
        Local_Energy=self.local_energy(atom_bond_fea)
        #print(Local_Energy)
        #local_energy=self.local_energy_softplus(Local_Energy)
#         local_energy=self.hidden_energy(local_energy)
#         local_energy=self.local_energy_softplus(local_energy)
        Voltage=self.pooling(Local_Energy, crystal_atom_idx)
        
        return Voltage, Local_Energy
    
#     def pooling(self, local_energy):
#         voltage=torch.mean(local_energy)
#         return voltage

    def pooling(self, local_energy, crystal_atom_idx):
            """
            Pooling the atom features to crystal features

            N: Total number of atoms in the batch
            N0: Total number of crystals in the batch

            Parameters
            ----------

            atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
              Atom feature vectors of the batch
            crystal_atom_idx: list of torch.LongTensor of length N0
              Mapping from the crystal idx to atom idx
            """
            assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
                local_energy.data.shape[0]
            summed_fea = [torch.mean(local_energy[idx_map], dim=0, keepdim=True)
                          for idx_map in crystal_atom_idx]
            return torch.cat(summed_fea, dim=0)
