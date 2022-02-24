import torch
import warnings
from .functions import coul_J, cutoff
from .eem_module import EEM

"""
Unit used here default
Length: Angstom
Charge: elementary charge +e
Energy: eV
"""

class SQEP(EEM):
    def __init__(self, f_cutoff='cos', units={'energy':'eV', 'length':"Angstrom"}, lower_bound=0.0):
        super().__init__(units=units, lower_bound=lower_bound)
        if callable(f_cutoff):
            self.f_cutoff = f_cutoff
        else:
            self.f_cutoff = cutoff(f_cutoff)
    
    def forward(self, species, coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, Ld, chi, Eext=None, q0=0.0 ):
        """
        species : shape (n_molecule, n_atom)
        coordinates : shape (n_molecule, n_atom, 3), unit Angstrom
        sigma : width, shape (n_molecule, n_atom), unit Angstrom
        K : shape (n_pair, 1), > 0.0, from hippynn.layers.targets.HBondSymmetric
        Ld : diagonal part of L, as in HIPNN, there is no pair between same atom. shape (n_real_atom,1)
        chi : presented in E_{corr}, shape (n_real_atom,1), from hippynn.layers.targets.HCharge (with n_target=1)
        Eext: a callable take r as input and return phi for each atom in each molecule, or a 3D vector for each molecule (n_molecule, 3)

        real_atoms: check chemnn.layers.indexers.PaddingIndexer, shape (n_molecule*n_atom)
            # eg. water:          [[8,1,1,0],[8,1,1,0]]
            #     nonblank:       [[T,T,T,F],[T,T,T,F]]
            #     real_atoms:     [0,1,2,4,5,6]
            #     inv_real_atoms: [0,1,2,0,3,4,5,0]
            #     pair_first:     [0,0,1,2,3,3,4,5]
            #     pair_second:    [1,2,0,0,4,5,3,3]
            #     mol_id:         [0,0,0,0,1,1,1,1]
            #     real_atom[pair_first]: [0,0,1,2,4,4,5,6]
        pair_first, pair_second, pair_dist: check chemnn.layers.targets.HBondSymmetric
        pair_first, pair_second, atom index for each pair, shape (n_pairs, ), only need pair_first < pair_second
        pair_dist: pair distance

        q0 : bare charge, shape (n_molecule, n_atom), unit +e, if None or 0.0, treat as 0.0
        q_mol : net charge for molecue, a float number of tensor with shape (n_moldule, ), right now ignore, or treat as 0.0 #TODO 
        cells : periodic cells 

        # formulas:
        i, j : atom
        \mu, \nu : bond
        E_{coul} = \frac{1}{2} \sum_{i,j} K_ij * p_{ij}^2 + \frac{1}{2} \sum_{i,j} q_i J_{ij} q_j + \sum_i \phi_{ext,i} q_i
        J_{ij} = (1-\delta_{ij}) * \frac{1}{4\pi\epsilon_0 r_{ij}} *
                 {erf}(\frac{r_{ij}}{\sqrt{\sigma_i^2+\sigma_j^2}})
        
        E_{corr} = \frac{1}{2} \sum_{\mu, \nu} p_{\mu} K_{\mu \nu} p_{\nu} + \
                   \frac{1}{2} \sum_{i,j} q_i L_{ij} q_j + \
                   \sum_i \chi_i q_i 

        """

        dtype = coordinates.dtype
        device = coordinates.device
        n_molecule, n_atom = coordinates.shape[:2]
        nonblank = species>0

        sigma0 = torch.zeros(species.shape, dtype=dtype, device=device)
        sigma0[nonblank] = sigma.reshape(-1)
        sigma = self.sp_L(sigma0)

        chi0 = torch.zeros(species.shape, dtype=dtype, device=device)
        chi0[nonblank] = chi.reshape(-1)
        chi = chi0.unsqueeze(2)
        
        n_real_atom = real_atoms.shape[0]
        atom_mol_id = real_atoms // n_atom
        
        Ld = self.sp_E(Ld) + self.c
        Ld0 = torch.zeros(species.shape, dtype=dtype, device=device)
        Ld0[nonblank] = Ld.reshape(-1)
        Ld = Ld0

        cond = pair_first < pair_second
        pair_first = pair_first[cond]
        pair_second = pair_second[cond]
        K = self.sp_E(K) 
        #K = self.sp_E(K[cond]) 

        # phi : potential for atom i, shape (n_molecule, n_atom)
        if Eext == None:
            Eext = torch.zeros(n_molecule, 3, dtype=dtype, device=device, requires_grad=True)
        else:
            assert Eext.dim() == 2, "Eext is not implemented, and not used"
            # Eext is a vector for each molecule, shape (n_molecule, 3)
            Eext.requires_grad_(True)

        phi = -torch.sum(Eext.unsqueeze(1)*coordinates, dim=2).unsqueeze(2)
        
        J = coul_J(nonblank, coordinates, sigma) * self.conversion_factor

        f_cut_r = self.f_cutoff(pair_dist[cond])

        pair_mol_id = real_atoms[pair_first] // n_atom
        mol_pairs = torch.zeros(n_molecule, dtype=torch.int64, device=device)
        mol_pairs.index_add_(0, pair_mol_id, torch.ones_like(pair_mol_id))
        max_pair = mol_pairs.max().item()
        A = torch.zeros(n_molecule, max_pair, max_pair, dtype=dtype, device=device)
        S = torch.zeros(n_molecule, n_atom, max_pair, dtype=dtype, device=device)
        
        nonblank_pair = torch.zeros(n_molecule, max_pair, dtype=torch.bool, device=device)
        for i in range(n_molecule):
            nonblank_pair[i,:mol_pairs[i]] = True
        nonblank_pair = nonblank_pair.reshape(-1)
        pair_indx = torch.arange(n_molecule*max_pair, dtype=torch.int64, device=device)[nonblank_pair]
        pair_indx_mol = pair_indx%max_pair
        A[pair_mol_id, pair_indx_mol, pair_indx_mol] = K.reshape(-1)[cond] #*f_cut_r**2

        """
        v = (k, l)
        S_{i, v} = fcut(r_kl) * +1 if i=l
                              * -1 if i=k
        """
        real_pair_first_n_atom  = real_atoms[pair_first] % n_atom
        real_pair_second_n_atom = real_atoms[pair_second] % n_atom
        S[pair_mol_id, real_pair_first_n_atom, pair_indx_mol] += -f_cut_r
        S[pair_mol_id, real_pair_second_n_atom, pair_indx_mol] += f_cut_r
        
        LJ = torch.diag_embed(Ld) + J

        STLJdS = torch.matmul(S.transpose(1,2),torch.matmul(LJ, S))
        A = A + STLJdS
        # padding
        indx = torch.arange(max_pair, dtype=torch.int64, device=device)
        for i in range(n_molecule):
            A[i,indx[mol_pairs[i]:], indx[mol_pairs[i]:]] = 1.0
       
        # b = - S^T ((L+J) q^0 + (chi+phi))
        # c = 0.5 * q^0^T (L+J) q^0 + q^0^T (chi+ phi)

        if torch.is_tensor(q0):
            q0 = q0.unsqueeze(2)
            LJq0 = torch.matmul(LJ, q0) 
            c = torch.matmul(q0.transpose(1,2), 0.5*LJq0 + chi + phi)
        else:
            LJq0 = 0.0
            c = 0.0

        b = - torch.matmul(S.transpose(1,2), LJq0+chi+phi)
        
        p  =  torch.linalg.solve(A, b)
        
        # Etot =  0.5 p^T A p - p^T b + c
        # Ap = b
        # Etot = -0.5 p^T b + c

        Etot =  -0.5*torch.matmul(p.transpose(1,2), b) + c #.reshape(-1,1)
        q = q0 + torch.matmul(S, p)

        d = self.dipole(q, coordinates)
        quadrupole = self.quadrupole(q, species, coordinates)
        # alpha = self.polarizability(A, S, coordinates)
        return q.reshape(n_molecule,n_atom), Etot.reshape(n_molecule,1), d, quadrupole, Eext #, alpha

    # @staticmethod
    # def polarizability(A, S, coordinates):
    #     """
    #     A : shape (n_molecule, max_pair, max_pair) 
    #         the one used in forward() torch.solve
    #     S : shape (n_molecule, )
    #     coordiantes : shape (n_molecule, n_atom, 3)
    #     tensor alpha_{\mu \nu} = - r_{\mu}^T * S A^{-1} S^T * r_{\nu}
    #     """
    #     ST_R = torch.matmul(S.transpose(1,2),coordinates)

    #     A_inv_ST_R, _ = torch.solve(ST_R, A) 
        
    #     return -torch.matmul(ST_R.transpose(1,2), A_inv_ST_R)     
