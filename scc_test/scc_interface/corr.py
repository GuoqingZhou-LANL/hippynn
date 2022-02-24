from typing import ForwardRef
import torch
from hippynn.layers import indexers

class MolTensor(torch.nn.Module):
    """
    Compute intrisic quadrupole or polarizability for each molecule
    """
    def __init__(self,feature_sizes,first_is_interacting=False):
        super().__init__()
        self.first_is_interacting = first_is_interacting
        if first_is_interacting:
          feature_sizes = feature_sizes[1:]
        self.feature_sizes = feature_sizes
        self.summer = indexers.MolSummer()
        self.n_terms = len(feature_sizes)
        biases = (first_is_interacting,*(True for _ in range(self.n_terms-1)))
        self.layers = torch.nn.ModuleList(torch.nn.Linear(nf,1,bias=bias)
                                          for nf,bias in zip(feature_sizes,biases))

    def forward(self,all_features,mol_index,n_molecules):
        """
        Pytorch Enforced Forward function

        :param: all_features a list of feature tensors:
        :param: mol_index the molecular index for atoms in the batch
        :param: total number of molecules in the batch
        :return: intrisic tensor (quadrupole or polarizability) for each molecule
        """
        if self.first_is_interacting:
            all_features = all_features[1:]
        partial = [lay(x) for x,lay in zip(all_features,self.layers)]
        total_atom = sum(partial)
        total_mol = self.summer(total_atom,mol_index,n_molecules)
        # embedding as a tensor
        identity = torch.eye(3,dtype=total_mol.dtype, device=total_mol.device)
        return total_mol.unsqueeze(2)*identity.unsqueeze(0)

class AlphaE(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, alpha, Eext):
        return torch.matmul(alpha, Eext.unsqueeze(2)).reshape(-1,3)

class AlphaEE(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, alpha, Eext):
        return torch.matmul(Eext.unsqueeze(1), torch.matmul(alpha, Eext.unsqueeze(2))).reshape(-1,1)