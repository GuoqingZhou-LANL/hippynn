import torch

from hippynn.layers.indexers import OneHotSpecies, PaddingIndexer
from hippynn.layers.pairs import OpenPairIndexer
from hippynn.networks.hipnn import Hipnn, HipnnVec

class MyModel(torch.nn.Module):

    def __init__(self, network_params={}):
        super().__init__()
        self.network_params = self.update_network_params(network_params)
        #
        self.paddingindexer = PaddingIndexer()
        
        self.hipnn = Hipnn(**self.network_params)
        #self.hipnn = HipnnVec(**self.network_params)
        self.onehot = OneHotSpecies(species_set=self.hipnn.species_set)
        self.pairindexer = OpenPairIndexer(hard_dist_cutoff=self.hipnn.dist_hard_max)
        self.predict = PredictAtom(feature_sizes=self.hipnn.feature_sizes, n_target=1)
    
    def update_network_params(self, network_params):
        params = {
        "possible_species": [0,1,8],   # Z values of the elements
        'n_features': 40,                     # Number of neurons at each layer
        "n_sensitivities": 20,                # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.7,  # qm7 1.7  qm9 .85  AL100 .85
        "dist_soft_max": 5.,  # qm7 10.  qm9 5.   AL100 5.
        "dist_hard_max": 7.5,  # qm7 15.  qm9 7.5  AL100 7.5
        "n_interaction_layers": 1,            # Number of interaction blocks
        "n_atom_layers": 3,                   # Number of atom layers in an interaction block
        }
        params.update(network_params)
        return params
    
    def regularization_params(self):
        params = self.hipnn.regularization_params() + self.predict.regularization_params
        return params
    
    def forward(self, species, coordinates):
        encoding, nonblank = self.onehot(species)
        indexed_features, real_atoms, inv_real_atoms, mol_index,atom_index,n_molecules, n_atoms_max \
             = self.paddingindexer(encoding, nonblank)
        pair_dist, pair_first, pair_second, pair_coord = \
             self.pairindexer(coordinates, nonblank, real_atoms, inv_real_atoms)
        output_features = self.hipnn(indexed_features, pair_first, pair_second, pair_dist)
        #output_features = self.hipnn(indexed_features, pair_first, pair_second, pair_dist, pair_coord)
        print(output_features[0].shape, output_features[1].shape)
        raise
        for f in output_features:
            print(f.shape, f.dtype)
        output = self.predict(output_features)
        return output


class PredictAtom(torch.nn.Module):
    """
    Compute the Charge
    """
    def __init__(self,feature_sizes,first_is_interacting=False,n_target=1):
        super().__init__()
        self.feature_sizes = feature_sizes

        self.n_terms = len(feature_sizes)
        self.n_target = n_target
        biases = (first_is_interacting,*(True for _ in range(self.n_terms-1)))
        self.layers = torch.nn.ModuleList(torch.nn.Linear(nf,n_target,bias=bias)
                                          for nf,bias in zip(feature_sizes,biases))
    
    def regularization_params(self):
        params = []
        for lay in self.layers:
            params.append(lay.weight)

    def forward(self,all_features):
        """
        :param all_features a list of feature tensors:
        :return: predict n_target for each atom
        """
        output = sum([lay(x) for x,lay in zip(all_features,self.layers)])

        return output

if __name__ == '__main__':
    dtype=torch.float64
    device=torch.device("cuda")
    torch.set_default_dtype(dtype)
    import numpy as np
    import hippynn 
    hippynn.custom_kernels.set_custom_kernels (False)
    m = MyModel().to(device)
    N = 2
    R = torch.tensor(np.load("R.npy")[:N], device=device) #, dtype=torch.long)
    Z = torch.tensor(np.load("Z.npy")[:N], device=device) #, dtype=torch.double)
    f = m(Z, R)
    print(f, Z)

