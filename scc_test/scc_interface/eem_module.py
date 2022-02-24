# EEM: Electron Equilibrium Method
# Guoqing Zhou (guoqingz@lanl.gov)

import torch
import warnings
from .functions import coul_J, E_h, mass

class EEM(torch.nn.Module):

    def __init__(self, units={'energy':'eV', 'length':"Angstrom"}, lower_bound=0.0):
        """
        default unit used here 
        Length: Angstom
        Charge: elementary charge +e
        Energy: eV
        lower_bound : a hard bound for Ld (diagonal element of L matrix)
        """
        super().__init__()   
        
        self.units = units
        self.bound = lower_bound
        
        if self.units['length'] == 'Angstrom':
            length_scale = 1.0
        elif self.units['length'] == 'Bohr':
            length_scale = 0.529177210903
        else:
            raise ValueError("length with unit {} not supported yet".format(self.units['length']))

        if self.units['energy'] == 'eV':
            energy_scale = 1.0
        elif self.units['energy'] == 'Hartree':
            energy_scale = 27.211386245988
        elif self.units['energy'] == 'kcal/mol':
            energy_scale = 23.0609
        else:
            raise ValueError("energy with unit {} not supported yet".format(self.units['energy']))
        
        self.conversion_factor = 1.0/length_scale/energy_scale
        self.length_scale = length_scale
        self.energy_scale = energy_scale

        self.c = self.bound * E_h / energy_scale

        self.sp_L = torch.nn.Softplus(beta=5.0*length_scale) # used for enforce sigma to be positive, L: length
        self.sp_E = torch.nn.Softplus(beta=5.0*energy_scale) # used for enforce Ld to be positive, E: energy
        
     
    def forward(self, species, coordinates, sigma, Ld, chi, Eext=None, q_mol=None):
        """
        species : shape (n_molecule, n_atom)
        coordinates : shape (n_molecule, n_atom, 3), unit Angstrom or bohr
        sigma : width, shape (n_real_atom,1), have same unit as coordinates
        Ld : diagonal part of L, as in HIPNN, there is no pair between same atom. shape (n_real_atom,1)
        chi : presented in E_{corr}, shape (n_molecule, n_atom), from chemnn.layers.targets.HCharge (with n_target=1)
        Eext: electric field, 3D vector for each molecule (n_molecule, 3),i.e. a constant electric field, if None then use 0

        q_mol : net charge for molecue, a float number of tensor with shape (n_moldule, ) 
        
        # formulas:
        i, j : atom
        \mu, \nu : bond
        E_{coul} = \frac{1}{2} \sum_{i,j} q_i J_{ij} q_j + \sum_i \phi_{ext,i} q_i
        J_{ij} = (1-\delta_{ij}) * \frac{1}{4\pi\epsilon_0 r_{ij}} *
                 {erf}(\frac{r_{ij}}{\sqrt{\sigma_i^2+\sigma_j^2}})
        
        E_{corr} = \frac{1}{2} \sum_{i} q_i L_{ii} q_i + \sum_i \chi_i q_i 

        """
        # make sure sigma is positive and is around 0.5
        dtype = coordinates.dtype
        device = coordinates.device
        nonblank = species > 0
        n_molecule, n_atom = coordinates.shape[:2]

        sigma0 = torch.zeros(species.shape, dtype=dtype, device=device)
        sigma0[nonblank] = sigma.reshape(-1)
        sigma = self.sp_L(sigma0)

        chi0 = torch.zeros(species.shape, dtype=dtype, device=device)
        chi0[nonblank] = chi.reshape(-1)
        chi = chi0
        
        # make sure Ld is positive
        Ld = self.sp_E(Ld) + self.c
        Ld0 = torch.zeros(species.shape, dtype=dtype, device=device)
        Ld0[nonblank] = Ld.reshape(-1)
        Ld = Ld0
     
        # phi : potential for atom i, shape (n_molecule, n_atom)
        if Eext == None:
            Eext = torch.zeros(n_molecule, 3, dtype=dtype, device=device, requires_grad=True)
        else:
            assert Eext.dim() == 2, "Eext is not implemented, and not used"
            # Eext is a vector for each molecule, shape (n_molecule, 3)
            Eext.requires_grad_(True)

        phi = -torch.sum(Eext.unsqueeze(1)*coordinates, dim=2) 

        J = coul_J(nonblank, coordinates, sigma)*self.conversion_factor
        
        # E  = 0.5 q^T * (L+J) * q + q^T * (chi+ phi)
        # dE/dq = 0 ==> (L+J)q = - chi - phi

        # have to replace the padding diagonal 0 with 1 on A, shift A by 1.0
        #A = L_mat + J
        A0 = torch.zeros(n_molecule, n_atom+1, n_atom+1, dtype=dtype, device=device)
        # put diagonal elements of padding part as 1.0
        maskd = ((torch.arange(n_molecule,dtype=torch.int64,device=device)*(n_atom+1)**2).reshape((-1,1)) + \
            (torch.arange(n_atom,dtype=torch.int64,device=device)*(n_atom+2)).reshape((1,-1))).reshape(-1)[~nonblank.reshape(-1)]
        A0.reshape(-1)[maskd] = 1.0
        # add constraint
        maskc = ((torch.arange(n_molecule,dtype=torch.int64,device=device)*(n_atom+1)).reshape((-1,1)) + \
            (torch.arange(n_atom,dtype=torch.int64,device=device)).reshape((1,-1))).reshape(-1)[nonblank.reshape(-1)]
        A0.reshape(-1,n_atom+1)[maskc,-1] = 1.0
        A0[:,-1,:] = A0[:,:,-1]

        A = torch.diag_embed(Ld) + J
        A0[:,:-1,:-1] += A
        
        # q_mol : net charge, a float number or a tensor with shape (n_molecule, )
        # chi : shape (n_molecule, n_atom)
        # phi : same as chi
        if not torch.is_tensor(q_mol):
            q_mol = 0.0
        b = - (chi + phi)
        b0 =  torch.zeros(n_molecule, n_atom+1, dtype=dtype, device=device)
        b0[:,:-1] = b
        b0[:,-1] = q_mol

        q_tmp, _ = torch.solve(b0.unsqueeze(2), A0)
        q = q_tmp[:,:-1, :]
        # q: shape (n_molecule, n_atom, 1)
        Ecoul = 0.5*torch.matmul(q.transpose(1,2),torch.matmul(A,q)) + \
            torch.matmul(q.transpose(1,2), (chi+phi).unsqueeze(2))
        
        d = self.dipole(q, coordinates)
        quadrupole = self.quadrupole(q,species,coordinates)
        #alpha = self.polarizability(A0, coordinates)
    
        return q.reshape(n_molecule, n_atom), Ecoul.reshape(-1,1), d, quadrupole, Eext #, alpha
    
    @staticmethod
    def dipole(q, coordinates):
        """
        q : charge, shape (n_molecule, n_atom, 1)
        coordinates : shape (n_molecule, n_atom, 3)
        """
        return torch.sum(q*coordinates, dim=1)
    
    @staticmethod
    def quadrupole(q,species,coordinates):
        """
        q : charge, shape (n_molecule, n_atom, 1)
        coordinates : shape (n_molecule, n_atom, 3)
        species : shape (n_molecule, n_atom)
        """
        # remove center of mass
        masses = torch.tensor(mass, dtype=q.dtype, device=q.device)[species]
        R0 = (coordinates*masses.unsqueeze(2)).sum(dim=1, keepdim=True)/masses.sum(dim=1).reshape(-1,1,1)
        R = coordinates-R0
        quadrupole = ((q*R).unsqueeze(3)*R.unsqueeze(2)).sum(dim=1)
        # traceless form
        d = (quadrupole[...,0,0] + quadrupole[...,1,1] + quadrupole[...,2,2])/3.0
        for k in range(3):
            quadrupole[...,k,k] -= d
        return quadrupole

    # @staticmethod
    # def polarizability(A0, coordinates):
    #     """
    #     A0 : shape (n_molecule, n_atom+1, n_atom+1) 
    #         the one used in forward() torch.solve
    #     coordiantes : shape (n_molecule, n_atom, 3)
    #     tensor alpha_{\mu \nu} = - r_{\mu} * A^{-1} * r_{\nu}
    #     """
    #     # A0 includes constraint part in last row/column
    #     #Ainv_r, _ = torch.solve(coordinates, A0[:,:-1,:-1]) 
    #     #Ainv_e, _ = torch.solve(ones, A0[:,:-1,:-1]) 
    #     ones = torch.ones(coordinates.shape[0],coordinates.shape[1], 1, dtype=coordinates.dtype, device=coordinates.device)
    #     tmp = torch.cat([coordinates, ones], dim=-1)
    #     Ainv_tmp, _ = torch.solve(tmp, A0[:,:-1,:-1]) 
    #     Ainv_r = tmp[:,:,:3]
    #     Ainv_e = tmp[:,:,3:]
    #     eAinv_r = torch.matmul(ones.transpose(1,2), Ainv_r) # shape (n_molecule, 1, 3)
    #     eAinv_e = torch.matmul(ones.transpose(1,2), Ainv_e) # shape (n_molecule, 1, 1)     
    #     return -torch.matmul(coordinates.transpose(1,2), Ainv_r) +\
    #          torch.matmul(eAinv_r.transpose(1,2), eAinv_r)/eAinv_e
