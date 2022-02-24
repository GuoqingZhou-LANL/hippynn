import torch

from hippynn.layers.indexers import OneHotSpecies, PaddingIndexer
from hippynn.layers.pairs import OpenPairIndexer
from hippynn.networks.hipnn import Hipnn, HipnnVec
from hippynn.layers.hiplayers import InteractLayer, SensitivityModule, InverseSensitivityModule


#TODO n_target is not used, modify to accomadate multiple target cases


class LSTMModel(torch.nn.Module):
    """
    A LSTM few to many model
    """

    def __init__(self, network0_params={}, network1_params={'possible_species':[0,1]}, n_in=2, n_out=3, n_hist = 3, n_target=1):
        super().__init__()
        """
        network0_params : parameters for first HIPNN which is used to get h_0, c_0
        network1_params : parameters for second HIPNN which is used to parse raw input x_t
        use n_in (i.e. x_0, x_1, ..., x_(n_in-1)) inputs 
            to predict n_out outputs, i.e. y_(n_in), ... y(n_in + n_out - 1)
        n_target : output quantity shape (atom, n_target) at time t
        """
        # two ways to put n_in input, one is concatenated to onehot in hipnn0
        # second is to do as normal RNN network, as a few to many RNN, ignore the output from first n_in-1
        self.n_in = n_in
        self.n_out = n_out
        self.n_hist = n_hist
        self.n_target = n_target
        self.network0_params = self.update_network_params(network0_params) 
        self.network0_params["possible_species"] += [1 for _ in range(n_hist)]
        # used for initial hidden and cell states
        self.network0_params['n_features'] *= 2
        # half for h_0, half for c_0

        self.network1_params = self.update_network_params(network1_params) 
        assert self.network0_params['n_features'] == self.network1_params['n_features']*2, "n_features are not consistent for two HIPNN network"
        # used for parse raw input at t (which is mulliken charge or local spin, etc) to generate input x_t

        self.paddingindexer = PaddingIndexer()
        
        self.hipnn0 = Hipnn(**self.network0_params)
        # hipnn0 output features from last layer (one interacting layer) are used as h_0, c_0

        self.hipnn1 = Hipnn(**self.network1_params)
        
        #self.hipnnvec = HipnnVec(**self.network_params)
        self.onehot = OneHotSpecies(species_set=self.hipnn0.species_set[:-n_hist])
        self.pairindexer = OpenPairIndexer(hard_dist_cutoff=self.hipnn0.dist_hard_max)

        # take the first output of hipnn0 and output y_t from LSTM to get predicted yhat_t 
        feature_sizes = (self.hipnn0.feature_sizes[0], self.hipnn0.feature_sizes[-1]//2)
        self.predict = PredictAtom(feature_sizes=feature_sizes, n_target=n_target)
        
        # LSTM components
        # determine sensitivity layer
        sensitivity_type = self.network0_params['sensitivity_type']
        if sensitivity_type == "inverse":
            sensitivityModule = InverseSensitivityModule
        elif sensitivity_type == "linear":
            # linear default
            sensitivityModule = SensitivityModule
        elif callable(sensitivity_type):
            sensitivityModule = sensitivity_type
        else:
            raise TypeError("Invalid sensitivity type:",sensitivity_type)
    
        # intlayer_params = (self.hipnn1.nf, self.hipnn1.nf, self.hipnn1.n_sensitivities, \
        #     self.hipnn1.dist_soft_min, self.hipnn1.dist_soft_max, self.hipnn1.dist_hard_max, \
        #     sensitivityModule)
        # self.W = torch.nn.ModuleList(InteractLayer(*intlayer_params) for _ in range(8))

        intlayer_params = (self.hipnn1.nf*2, self.hipnn1.nf*4, self.hipnn1.n_sensitivities, \
            self.hipnn1.dist_soft_min, self.hipnn1.dist_soft_max, self.hipnn1.dist_hard_max, \
            sensitivityModule)
        self.W = InteractLayer(*intlayer_params)

    def update_network_params(self, network_params):
        params = {
        "possible_species": [0,1,8],   # Z values of the elements
        'n_features': 20,                     # Number of neurons at each layer
        "n_sensitivities": 20,                # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.7,  # qm7 1.7  qm9 .85  AL100 .85
        "dist_soft_max": 5.,  # qm7 10.  qm9 5.   AL100 5.
        "dist_hard_max": 7.5,  # qm7 15.  qm9 7.5  AL100 7.5
        "n_interaction_layers": 1,            # Number of interaction blocks
        "n_atom_layers": 3,                   # Number of atom layers in an interaction block
        "sensitivity_type" : "inverse",
        }
        params.update(network_params)
        return params
    
    def regularization_params(self):
        params = self.hipnn0.regularization_params() + \
            self.hipnn1.regularization_params() + \
            self.predict.regularization_params()
        params.append(self.W.int_weights)
        params.append(self.W.selfint.weight)
        return params
    
    def onestep(self, x_raw_enc, h_t, c_t, pair_first, pair_second, pair_dist):
        x_t = self.hipnn1(x_raw_enc, pair_first, pair_second, pair_dist)[-1]

        # i_t = torch.sigmoid(self.W[0](x_t, pair_first, pair_second, pair_dist) + self.W[1](h_t, pair_first, pair_second, pair_dist))
        # f_t = torch.sigmoid(self.W[2](x_t, pair_first, pair_second, pair_dist) + self.W[3](h_t, pair_first, pair_second, pair_dist))
        # g_t = torch.tanh(self.W[4](x_t, pair_first, pair_second, pair_dist) + self.W[5](h_t, pair_first, pair_second, pair_dist))
        # o_t = torch.sigmoid(self.W[6](x_t, pair_first, pair_second, pair_dist) + self.W[7](h_t, pair_first, pair_second, pair_dist))
        # c_t = f_t * c_t + i_t * g_t
        # h_t = o_t * torch.tanh(c_t)

        tmp = self.W(torch.cat([x_t, h_t], dim=-1), pair_first, pair_second, pair_dist).reshape(x_t.shape[0],-1, 4)
        o_t = torch.sigmoid(tmp[...,:,3])
        c_t = torch.sigmoid(tmp[...,:,1]) * c_t + torch.sigmoid(tmp[...,:,0]) * torch.tanh(tmp[...,:,2])
        h_t = o_t * torch.tanh(c_t)

        return o_t, h_t, c_t
    
    def forward(self, species, coordinates, x_h, x_raw, n_out=-1):
        """
        x : per atom local property, like mulliken charge, or local spin etc, default shape (n_mol,n_atom, n_in)
            raw inputs from t=0 to t=n_in-1
        """
        if n_out == -1:
            n_out = self.n_out
        else:
            assert n_out>0, "n_out must be positive"
        encoding, nonblank = self.onehot(species)
        indexed_features, real_atoms, inv_real_atoms, mol_index,atom_index,n_molecules, n_atoms_max \
             = self.paddingindexer(encoding, nonblank)
        pair_dist, pair_first, pair_second, pair_coord = \
             self.pairindexer(coordinates, nonblank, real_atoms, inv_real_atoms)

        ind_f = torch.cat([indexed_features, x_h[nonblank]],dim=1)

        output_features = self.hipnn0(ind_f, pair_first, pair_second, pair_dist)
        c_t = output_features[-1][:,:self.hipnn0.feature_sizes[-1]//2]
        h_t = output_features[-1][:,(self.hipnn0.feature_sizes[-1]//2):]
        x_raw_enc = x_raw[nonblank]

        for t in range(self.n_in):
            o_t, h_t, c_t = self.onestep(x_raw_enc[:,t:(t+1)], h_t, c_t, pair_first, pair_second, pair_dist)

        x_raw_enc0 = self.predict((output_features[0], o_t))

        output = []
        for t in range(n_out):
            o_t, h_t, c_t = self.onestep(x_raw_enc0, h_t, c_t, pair_first, pair_second, pair_dist)
            x_raw_enc0 = self.predict((output_features[0], o_t))
            output.append(x_raw_enc0)

        return torch.cat(output, dim=-1), nonblank

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
        return params

    def forward(self,all_features):
        """
        :param all_features a list of feature tensors:
        :return: predict n_target for each atom
        """
        output = sum([lay(x) for x,lay in zip(all_features,self.layers)])

        return output



from hippynn.graphs.nodes.inputs import PositionsNode, SpeciesNode
from hippynn.graphs.indextypes import IdxType
from hippynn.graphs.nodes.base import MultiNode, AutoKw, SingleNode

class LSTMNode(AutoKw, MultiNode):
    _input_names = "species", "coordinates", "x_hist", "x_raw_t"
    _output_names = "x_output_t", "nonblank"
    _main_output = "x_output_t"
    _child_index_states = IdxType.Atoms, IdxType.MolAtom
    _auto_module_class = LSTMModel
    
    def __init__(self, name, parents, network0_params={}, network1_params={'possible_species':[0,1]}, n_in=2, n_out=3, n_hist=2, module='auto', **kwargs):
        self.module_kwargs = dict(network0_params=network0_params, network1_params=network1_params, n_in=n_in, n_out=n_out, n_hist=n_hist)
        super().__init__(name, parents, module=module, **kwargs)


class mask(torch.nn.Module):
    def forward(self, x, mask):
        if x.shape[0] == mask.shape[0]:
            return x[mask]
        else:
            return x

class MaskNode(AutoKw, SingleNode):
    _input_names = "x", "mask"
    _index_state = IdxType.NotFound
    _auto_module_class = mask

    def __init__(self,name,parents,module='auto',**kwargs):
        self.module_kwargs = {}
        super().__init__(name,parents,module=module,**kwargs)


if __name__ == '__main__':
    dtype=torch.float64
    device=torch.device("cuda")
    torch.set_default_dtype(dtype)
    import numpy as np
    import hippynn 
    hippynn.custom_kernels.set_custom_kernels (False)
    m = LSTMModel().to(device)
    R = torch.tensor(np.load("R0.npy"), device=device) #, dtype=torch.long)
    Z = torch.tensor(np.load("Z0.npy"), device=device) #, dtype=torch.double)
    x = torch.tensor(np.load('x0.npy'), device=device) #, dtype=torch.double)
    xh = torch.rand(2,6,3, device=device)
    output, nonblank = m(Z, R, xh, x)
    
    y = torch.rand(2,6,3, device=device)
    #print(output.shape, y.shape, nonblank.shape)
    l2 = hippynn.layers.regularization.LPReg(m)
    L = torch.sum((output-y[nonblank])**2) + l2().sum()
    L.backward()
    m.eval()
    k,_ = m(Z[0:1], R[0:1], xh[0:1], x[0:1], n_out=4)
    print(k)
    #print(L.shape)
    #print(f.shape, Z)

