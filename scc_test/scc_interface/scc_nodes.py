import torch

from .eem_module import EEM
from .sqe_module import SQE
from .sqep_module import SQEP
from .functions import polarizability, quadrupole

from hippynn.graphs.nodes.base import MultiNode, AutoKw, find_unique_relative, ExpandParents, SingleNode
from hippynn.graphs.indextypes import IdxType
from hippynn.graphs.nodes.networks import Network
from hippynn.graphs.nodes.targets import HChargeNode, HBondNode
from hippynn.graphs.nodes.inputs import PositionsNode, SpeciesNode, CellNode
from hippynn.graphs.nodes.pairs import PairIndexer
from hippynn.graphs.nodes.indexers import PaddingIndexer

class EEMNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "species", "coordinates", "sigma", "Ld", "chi"
    _output_names = "charge", "coul_energy", "dipole", "quadrupole", "external_field"
    _main_output = "charge"
    _output_index_states = (IdxType.Molecules, )*len(_output_names)
    _auto_module_class = EEM

    @_parent_expander.match(Network)
    def expand0(self, network, **kwargs):
        sigma = HChargeNode("SCC_sigma", network, 
                    module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("SCC_chi", network, 
                    module_kwargs=dict(first_is_interacting=False))
        Ld = HChargeNode("SCC_Ld", network,
                    module_kwargs=dict(first_is_interacting=False))
        return sigma, chi, Ld

    @_parent_expander.match(HChargeNode, HChargeNode, HChargeNode)
    def expand1(self, sigma, chi, Ld, **kwargs):
        positions = find_unique_relative([sigma, chi], PositionsNode)
        species = find_unique_relative([sigma, chi], SpeciesNode)
        #indexer = find_unique_relative([sigma, chi], PaddingIndexer)
        #pairs = find_unique_relative([sigma, chi], PairIndexer)
        
        return species, positions, sigma.main_output, Ld.main_output, chi.main_output

    def __init__(self, name, parents, lower_bound=0.0, \
                 units={'energy':'eV', 'length':"Angstrom"}, module='auto', **kwargs):
        parents = self.expand_parents(parents, **kwargs)
        self.module_kwargs = dict(lower_bound=lower_bound, units=units)
        super().__init__(name, parents, module=module, **kwargs)

class SQENode(ExpandParents, AutoKw, MultiNode):
    _input_names = "species", "coordinates", "sigma", "real_atoms", \
                   "pair_first", "pair_second", "pair_dist", "K", "Ld", "chi"
    _output_names = "charge", "coul_energy", "dipole", "quadrupole", "external_field"
    _main_output = "charge"
    _output_index_states = (IdxType.Molecules, )*len(_output_names)
    _auto_module_class = SQE

    @_parent_expander.match(Network)
    def expand0(self, network, **kwargs):
        netmod = network.torch_module
        bond_parameters = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "symmetric" : True,
            "positive" : True,
            "n_target":1,
            "all_pairs":False,
        }
        sigma = HChargeNode("SQE_sigma", network, 
                    module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("SQE_chi", network, 
                    module_kwargs=dict(first_is_interacting=False))
        K = HBondNode("SQE_K", network, module_kwargs = bond_parameters)
        Ld = HChargeNode("SQE_Ld", network,
                    module_kwargs=dict(first_is_interacting=False))
        return sigma, chi, K, Ld

    @_parent_expander.match(Network, Network)
    def expand1(self, network1, network2, **kwargs):
        netmod = network2.torch_module
        bond_parameters = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "symmetric" : True,
            "positive" : True,
            "n_target":1,
            "all_pairs":False,
        }
        sigma = HChargeNode("SQE_sigma", network1, 
                    module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("SQE_chi", network1, 
                    module_kwargs=dict(first_is_interacting=False))
        K = HBondNode("SQE_K", network2, module_kwargs = bond_parameters)
        Ld = HChargeNode("SQE_Ld", network1,
                    module_kwargs=dict(first_is_interacting=False))
        return sigma, chi, K, Ld

    @_parent_expander.match(HChargeNode, HChargeNode, HBondNode, HChargeNode)
    def expand2(self, sigma, chi, K, Ld, **kwargs):
        positions = find_unique_relative([sigma, chi], PositionsNode)
        species = find_unique_relative([sigma, chi], SpeciesNode)
        indexer = find_unique_relative([sigma, chi], PaddingIndexer)
        pairs = find_unique_relative([sigma, chi], PairIndexer)
        
        return species, positions, sigma.main_output, indexer.real_atoms, pairs.pair_first, \
               pairs.pair_second, pairs.pair_dist, K.main_output, Ld.main_output, chi.main_output

    def __init__(self, name, parents, f_cutoff='cos', lower_bound=0.01, \
                 units={'energy':'eV', 'length':"Angstrom"}, module='auto', **kwargs):
        parents = self.expand_parents(parents, **kwargs)
        self.module_kwargs = dict(f_cutoff=f_cutoff, lower_bound=lower_bound, units=units)
        super().__init__(name, parents, module=module, **kwargs)

class SQEPNode(SQENode):
    _auto_module_class = SQEP

class PolarNode(AutoKw, SingleNode):
    _input_names = "dipole","external_field"
    _index_state = IdxType.Molecules
    _auto_module_class = polarizability
    def __init__(self,name,parents,module='auto',**kwargs):
        dipole, Eext = parents
        Eext.requires_grad = True
        self.module_kwargs = {}
        super().__init__(name,parents,module=module,**kwargs)

class QuadruNode(AutoKw, SingleNode):
    _input_names = "charge", "species", "positions"
    _index_state = IdxType.Molecules
    _auto_module_class = quadrupole
    def __init__(self,name,parents, traceless=True, module='auto',**kwargs):
        self.module_kwargs = {"traceless":traceless}
        super().__init__(name,parents,module=module,**kwargs)

from hippynn.graphs.nodes.base import InputNode

class ExternalFieldNode(InputNode):
    _index_state = IdxType.Molecules
    input_type_str = "external_field"

class EEMEFNode(EEMNode):
    _input_names = "species", "coordinates", "sigma", "Ld", "chi", "Eext"

    @_parent_expander.match(Network, ExternalFieldNode)
    def expand0(self, network, Eext, **kwargs):
        sigma = HChargeNode("SCC_sigma", network, 
                    module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("SCC_chi", network, 
                    module_kwargs=dict(first_is_interacting=False))
        Ld = HChargeNode("SCC_Ld", network,
                    module_kwargs=dict(first_is_interacting=False))
        return sigma, chi, Ld, Eext
    
    @_parent_expander.match(HChargeNode, HChargeNode, HChargeNode, ExternalFieldNode)
    def expand1(self, sigma, chi, Ld, Eext, **kwargs):
        positions = find_unique_relative([sigma, chi], PositionsNode)
        species = find_unique_relative([sigma, chi], SpeciesNode)
        #indexer = find_unique_relative([sigma, chi], PaddingIndexer)
        #pairs = find_unique_relative([sigma, chi], PairIndexer)
        
        return species, positions, sigma.main_output, Ld.main_output, chi.main_output, Eext


class SQEEFNode(SQENode):
    _input_names = "species", "coordinates", "sigma", "real_atoms", \
                   "pair_first", "pair_second", "pair_dist", "K", "Ld", "chi", "Eext"
    
    @_parent_expander.match(Network, ExternalFieldNode)
    def expand0(self, network, Eext, **kwargs):
        netmod = network.torch_module
        bond_parameters = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "symmetric" : True,
            "positive" : True,
            "n_target":1,
            "all_pairs":False,
        }
        sigma = HChargeNode("SQE_sigma", network, 
                    module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("SQE_chi", network, 
                    module_kwargs=dict(first_is_interacting=False))
        K = HBondNode("SQE_K", network, module_kwargs = bond_parameters)
        Ld = HChargeNode("SQE_Ld", network,
                    module_kwargs=dict(first_is_interacting=False))
        return sigma, chi, K, Ld, Eext
    
    @_parent_expander.match(Network, Network, ExternalFieldNode)
    def expand1(self, network1, network2, Eext, **kwargs):
        netmod = network2.torch_module
        bond_parameters = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "symmetric" : True,
            "positive" : True,
            "n_target":1,
            "all_pairs":False,
        }
        sigma = HChargeNode("SQE_sigma", network1, 
                    module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("SQE_chi", network1, 
                    module_kwargs=dict(first_is_interacting=False))
        K = HBondNode("SQE_K", network2, module_kwargs = bond_parameters)
        Ld = HChargeNode("SQE_Ld", network1,
                    module_kwargs=dict(first_is_interacting=False))
        return sigma, chi, K, Ld, Eext
    
    @_parent_expander.match(HChargeNode, HChargeNode, HBondNode, HChargeNode, ExternalFieldNode)
    def expand2(self, sigma, chi, K, Ld, Eext, **kwargs):
        positions = find_unique_relative([sigma, chi], PositionsNode)
        species = find_unique_relative([sigma, chi], SpeciesNode)
        indexer = find_unique_relative([sigma, chi], PaddingIndexer)
        pairs = find_unique_relative([sigma, chi], PairIndexer)
        
        return species, positions, sigma.main_output, indexer.real_atoms, pairs.pair_first, \
               pairs.pair_second, pairs.pair_dist, K.main_output, Ld.main_output, chi.main_output, Eext

class SQEPEFNode(SQEEFNode):
    _auto_module_class = SQEP

from .corr import MolTensor, AlphaE, AlphaEE
from hippynn.graphs.nodes.tags import AtomIndexer

class MolTensorNode(AutoKw, ExpandParents, SingleNode):
    _input_names = "hier_features", "mol_index", "n_molecules"
    _index_state = IdxType.Molecules
    _auto_module_class = MolTensor

    @_parent_expander.match(Network)
    def expansion0(self, net, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
                self.module_kwargs["feature_sizes"]=net.torch_module.feature_sizes
        pdindexer = find_unique_relative(net, AtomIndexer)
        return net, pdindexer.mol_index, pdindexer.n_molecules

    def __init__(self, name, parents, first_is_interacting=False, module='auto',module_kwargs=None, **kwargs):
        self.module_kwargs = {"first_is_interacting": first_is_interacting}
        if module_kwargs is not None:
            self.module_kwargs = {**self.module_kwargs,**module_kwargs}
        parents = self.expand_parents(parents,**kwargs)
        super().__init__(name, parents, module=module, **kwargs)


class AlphaENode(AutoKw, SingleNode):
    _input_names = "alpha","external_field"
    _index_state = IdxType.Molecules
    _auto_module_class = AlphaE
    def __init__(self,name,parents,module='auto',**kwargs):
        self.module_kwargs = {}
        super().__init__(name,parents,module=module,**kwargs)

class AlphaEENode(AlphaENode):
    _auto_module_class = AlphaEE

from .dem_module import DEM

class DEMNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "species", "coordinates", "sigma", \
                   "pair_first", "pair_second", "pair_coord", "w0", "w1"
    _output_names = "dipole", "coul_energy", "quadrupole", "external_field", "dipole_atom"
    _main_output = "dipole"
    _output_index_states = (IdxType.Molecules, )*len(_output_names)
    _auto_module_class = DEM

    @_parent_expander.match(Network)
    def expand0(self, network, **kwargs):
        netmod = network.torch_module
        bond_parameters0 = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "positive" : False,
            "n_target":1,
        }
        bond_parameters1 = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "positive" : True,
            "n_target":1,
        }
        sigma = HChargeNode("DEM_sigma", network, 
                    module_kwargs=dict(first_is_interacting=False))
        w0 = HBondNode("DEM_w0", network, module_kwargs = bond_parameters0)
        w1 = HBondNode("DEM_w1", network, module_kwargs = bond_parameters1)
        return sigma, w0, w1

    @_parent_expander.match(HChargeNode, HBondNode, HBondNode)
    def expand2(self, sigma, w0, w1, **kwargs):
        positions = find_unique_relative(sigma, PositionsNode)
        species = find_unique_relative(sigma, SpeciesNode)
        indexer = find_unique_relative(sigma, PaddingIndexer)
        pairs = find_unique_relative(sigma, PairIndexer)
        
        return species, positions, sigma.main_output, pairs.pair_first, \
               pairs.pair_second, pairs.pair_coord, w0.main_output, w1.main_output

    def __init__(self, name, parents, lower_bound=0.5, \
                 units={'energy':'eV', 'length':"Angstrom"}, sigma_bound=0.0, module='auto', **kwargs):
        parents = self.expand_parents(parents, **kwargs)
        self.module_kwargs = dict(lower_bound=lower_bound, units=units, sigma_bound=sigma_bound)
        super().__init__(name, parents, module=module, **kwargs)


class DEMEFNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "species", "coordinates", "sigma", \
                   "pair_first", "pair_second", "pair_coord", "w0", "w1", "Eext"
    _output_names = "dipole", "coul_energy", "quadrupole", "external_field", "dipole_atom"
    _main_output = "dipole"
    _output_index_states = (IdxType.Molecules, )*len(_output_names)
    _auto_module_class = DEM

    @_parent_expander.match(Network, ExternalFieldNode)
    def expand0(self, network, Eext, **kwargs):
        netmod = network.torch_module
        bond_parameters0 = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "positive" : False,
            "n_target":1,
        }
        bond_parameters1 = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "positive" : True,
            "n_target":1,
        }
        sigma = HChargeNode("DEM_sigma", network, 
                    module_kwargs=dict(first_is_interacting=False))
        w0 = HBondNode("DEM_w0", network, module_kwargs = bond_parameters0)
        w1 = HBondNode("DEM_w1", network, module_kwargs = bond_parameters1)
        return sigma, w0, w1, Eext

    @_parent_expander.match(HChargeNode, HBondNode, HBondNode, ExternalFieldNode)
    def expand2(self, sigma, w0, w1, **kwargs):
        positions = find_unique_relative(sigma, PositionsNode)
        species = find_unique_relative(sigma, SpeciesNode)
        indexer = find_unique_relative(sigma, PaddingIndexer)
        pairs = find_unique_relative(sigma, PairIndexer)
        
        return species, positions, sigma.main_output, pairs.pair_first, \
               pairs.pair_second, pairs.pair_coord, w0.main_output, w1.main_output, Eext

    def __init__(self, name, parents, lower_bound=0.5, \
                 units={'energy':'eV', 'length':"Angstrom"}, sigma_bound=0.0, module='auto', **kwargs):
        parents = self.expand_parents(parents, **kwargs)
        self.module_kwargs = dict(lower_bound=lower_bound, units=units, sigma_bound=sigma_bound)
        super().__init__(name, parents, module=module, **kwargs)

from .functions import SplitMatrix
class SplitMatrixNode(AutoKw, MultiNode):
    _input_names = "var"
    _output_names = "diag", "offdiag"
    _main_output = "diag"
    _output_index_states = (IdxType.Molecules, )*len(_output_names)
    _auto_module_class = SplitMatrix

    def __init__(self,name,parents, traceless=False, module='auto',**kwargs):
        parents = (parents,)
        self.module_kwargs = dict(traceless=traceless)
        super().__init__(name,parents,module=module,**kwargs)

from .dem_pbc import DEM_PBC
from hippynn.graphs.nodes.pairs import PeriodicPairIndexer
from hippynn.graphs.nodes.tags import Encoder

class DEMPBCNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "species", "coordinates", "cells", "pair_first", "pair_coord", "w0", "w1", \
                   "real_atoms", "mol_index", "sigma", "pair_dist1", "pair_first1", "pair_second1", "pair_coord1"
    _output_names = "dipole", "coul_energy", "quadrupole", "external_field", "dipole_atom"
    _main_output = "dipole"
    _output_index_states = (IdxType.Molecules, )*len(_output_names)
    _auto_module_class = DEM_PBC

    @_parent_expander.match(Network)
    def expand0(self, network, coul_cutoff=15.0, **kwargs):
        positions = find_unique_relative(network, PositionsNode)
        species = find_unique_relative(network, SpeciesNode)
        cell = find_unique_relative(network, CellNode)
        encoder = find_unique_relative(network, Encoder)

        indexer = find_unique_relative(network, PaddingIndexer)
        pairs = find_unique_relative(network, PairIndexer)
        pidxer = find_unique_relative(network, PaddingIndexer, why_desc=None)

        netmod = network.torch_module
        bond_parameters0 = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "positive" : False,
            "n_target":1,
        }
        bond_parameters1 = {
            "n_dist":netmod.n_sensitivities,
            "dist_hard_max":netmod.dist_hard_max,
            "dist_soft_min":netmod.dist_soft_min,
            "dist_soft_max":netmod.dist_soft_max,
            "feature_sizes":netmod.feature_sizes,
            "positive" : True,
            "n_target":1,
        }
        sigma = HChargeNode("DEM_sigma", network, 
                    module_kwargs=dict(first_is_interacting=False))
        w0 = HBondNode("DEM_w0", network, module_kwargs = bond_parameters0)
        w1 = HBondNode("DEM_w1", network, module_kwargs = bond_parameters1)

        pairs1 = PeriodicPairIndexer("PairIndexer_Coul", (positions, encoder, pidxer, cell), dist_hard_max=coul_cutoff)

        return species, positions, cell, pairs.pair_first, pairs.coord, w0.main_output, w1.main_output, \
            indexer.real_atoms, indexer.mol_index, sigma.main_output,  pairs1.pair_dist, pairs1.pair_first, pairs1.pair_second, pairs1.coord

    def __init__(self, name, parents, lower_bound=0.5, \
                 units={'energy':'eV', 'length':"Angstrom"}, coul_cutoff=15.0, alpha=0.2, module='auto', **kwargs):

        parents = self.expand_parents(parents, coul_cutoff=coul_cutoff, **kwargs )
        self.module_kwargs = dict(lower_bound=lower_bound, units=units, coul_cutoff=coul_cutoff, alpha=alpha)
        super().__init__(name, parents, module=module, **kwargs)
