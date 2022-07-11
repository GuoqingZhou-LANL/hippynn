# DEM: Dipole Equilibrium Method
# Guoqing Zhou (guoqingz@lanl.gov)

import torch
import warnings
from .functions import e2_over_four_pi_epsilon_0, mass, a_0, E_h

class DEM(torch.nn.Module):

    def __init__(self, units={'energy':'eV', 'length':"Angstrom"}, lower_bound=0.5, sigma_bound = 0.0):
        """
        default unit used here 
        Length: Angstom
        Charge: elementary charge +e
        Energy: eV
        """
        super().__init__()   
        
        self.units = units
        self.lower_bound = lower_bound
        
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
        self.c = lower_bound / a_0 **3
        self.sigma_bound = sigma_bound


        self.sp_L = torch.nn.Softplus(beta=5.0*length_scale) # used for enforce sigma to be positive, L: length
        self.sp_K = torch.nn.Softplus(beta=1.0) # used for enforce K to be positive
        
     
    def forward(self, species, coordinates, sigma, pair_first, pair_second, pair_coord, w0, w1, Eext=None, q0=None):
        """
        species : shape (n_molecule, n_atom)
        coordinates : shape (n_molecule, n_atom, 3), unit Angstrom or bohr
        sigma : width, shape (n_real_atom,1), have same unit as coordinates
        chi : presented in E_{corr}, shape (n_molecule, n_atom, 3), from chemnn.layers.targets.HCharge (with n_target=1)
        Eext: electric field, 3D vector for each molecule (n_molecule, 3),i.e. a constant electric field, if None then use 0       
        # formulas:


        """
        # make sure sigma is positive and is around 0.5
        dtype = coordinates.dtype
        device = coordinates.device
        nonblank = species > 0
        n_molecule, n_atom = coordinates.shape[:2]
        n_real_atom = sigma.shape[0]

        rhat = pair_coord/torch.linalg.norm(pair_coord, dim=-1, keepdim=True)
        chi = torch.zeros((n_real_atom, 3), dtype=dtype, device=device)
        chi.index_add_(0,pair_first,w0*rhat)

        K = torch.zeros((n_real_atom, 3, 3), dtype=dtype, device=device)
        rt = rhat.unsqueeze(1)*rhat.unsqueeze(2)
        K.index_add_(0,pair_first, w1.unsqueeze(2)*rt)

        sigma0 = torch.zeros(species.shape, dtype=dtype, device=device)
        sigma0[nonblank] = sigma.reshape(-1)
        sigma = self.sp_L(sigma0) + self.sigma_bound
        

        chi0 = torch.zeros((n_molecule, n_atom, 3), dtype=dtype, device=device)
        chi0[nonblank] = chi
        chi = chi0

        K0 =  torch.zeros((n_molecule, n_atom, 3, 3), dtype=dtype, device=device)
        K0[nonblank] = K
        K = K0

        vec_0 = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)

        one = torch.tensor(1.0,dtype=dtype,device=device)
        zero = torch.tensor(0.0,dtype=dtype,device=device)
        mask = (nonblank.unsqueeze(1)*nonblank.unsqueeze(2))
        # diagonal 
        maskd = torch.eye(n_atom,device=device,dtype=torch.int).bool().unsqueeze(0)
        # put padding part of vec as 0
        vec_1 = torch.where(mask.unsqueeze(-1), vec_0, zero)

        #dis_0 = torch.linalg.norm(vec_1, dim=-1, keepdim=True)
        # put padding part as 1
        #dis_1 =  torch.where(mask.unsqueeze(-1), dis_0, one)
        # put diagonal part as 1
        #dis_2 = torch.where(maskd.unsqueeze(-1), one, dis_1)

        dis_0_sq = torch.sum(vec_1**2, dim=-1, keepdim=True)
        dis_1_sq =  torch.where(mask.unsqueeze(-1), dis_0_sq, one)
        dis_2_sq = torch.where(maskd.unsqueeze(-1), one, dis_1_sq)
        dis_2 = torch.sqrt(dis_2_sq)
        dis_1 = torch.where(maskd.unsqueeze(-1), zero, dis_2)
        dis_0 = torch.where(mask.unsqueeze(-1), dis_1, zero)
        
        L_ij = (3.0*vec_1.unsqueeze(-2)*vec_1.unsqueeze(-1) - \
            (dis_0**2).unsqueeze(-1)*torch.eye(3, dtype=dtype, device=device).reshape(1,1,1,3,3))/(dis_2**5).unsqueeze(-1)

        sigma2 = sigma**2
        ss0 = sigma2.unsqueeze(1)+sigma2.unsqueeze(2)
        ss = torch.where(mask, ss0, one)
        #print(dis.shape, ss.shape)
        f_ij = torch.erf( dis_0 / torch.sqrt( ss.unsqueeze(-1) ) )
        if q0 is not None:
            q0[species==0]=0.0
            bq = e2_over_four_pi_epsilon_0 * \
                (f_ij * q0.reshape(n_molecule, 1, n_atom, 1) * vec_1/dis_2**3).sum(dim=2).reshape(n_molecule, -1)
        else:
            c = 0.0
            bq = 0.0
        
        # phi : potential for atom i, shape (n_molecule, n_atom)
        if Eext == None:
            Eext = torch.zeros(n_molecule, 3, dtype=dtype, device=device, requires_grad=True)
        else:
            assert Eext.dim() == 2, "Eext is not implemented, and not used"
            # Eext is a vector for each molecule, shape (n_molecule, 3)
            Eext.requires_grad_(True)

        #A_ij = e2_over_four_pi_epsilon_0*( - f_ij*L_ij + K_ii )
        # A0 = torch.zeros(n_molecule, n_atom, n_atom, 3, 3, dtype=dtype, device=device)
        
        
        # put diagonal elements of the diagonal block of the padding part as 1.0
        # maskpd = ( (torch.arange(n_molecule,dtype=torch.int64,device=device)*n_atom**2).reshape((-1,1)) + \
        #            (torch.arange(n_atom,dtype=torch.int64,device=device)*(n_atom+1)).reshape((1,-1))
        #          ).reshape(-1)[~nonblank.reshape(-1)]
        # A0.reshape(-1,3,3)[maskpd] = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        # #A = e2_over_four_pi_epsilon_0 * ( torch.diag_embed(K.permute(0,2,3,1)).permute((0,3,4,1,2)) - f_ij.unsqueeze(-1)*L_ij ) + A0
        # A = e2_over_four_pi_epsilon_0 * ( - L_ij) + A0
        # seems like there is no need to have A0
        A = e2_over_four_pi_epsilon_0 * (
            ( torch.diag_embed(K.permute(0,2,3,1)).permute((0,3,4,1,2)) - f_ij.unsqueeze(-1)*L_ij ).permute((0,1,3,2,4) ).reshape(n_molecule, n_atom*3, n_atom*3) 
            + self.c * torch.eye(n_atom*3, dtype=dtype, device=device ).unsqueeze(0) )
        # A = e2_over_four_pi_epsilon_0 * (
        #     ( - f_ij.unsqueeze(-1)*L_ij ).permute((0,1,3,2,4) ).reshape(n_molecule, n_atom*3, n_atom*3) 
        #     + self.c * torch.eye(n_atom*3, dtype=dtype, device=device ).unsqueeze(0) )
        #print(A.reshape(-1)[maskpd])
        #A.reshape(-1)[maskpd] = 1.0
        #print(A.reshape(-1)[maskpd])
        
        #e, v = torch.linalg.eigh(A)
        #print(e.shape, v.shape)
        #print(e.min(dim=-1))

        b = (chi + Eext.unsqueeze(1)).reshape(n_molecule, -1) + bq
        p = torch.linalg.solve(A, b)
        #def check(grad):
        #    print(grad)
        #    #if torch.isnan(grad).any():
        #    #    print('nan in p grad')
        #    #    raise
        #p.register_hook(lambda grad:  check(grad))
        Ecoul = 0.5*torch.matmul(p.unsqueeze(1), torch.matmul(A, p.unsqueeze(2))) - torch.matmul(b.unsqueeze(1), p.unsqueeze(2))
        #Ecoul = (p**2).sum(dim=-1, keepdim=True)
        #mf = MyFunc.apply
        #Ecoul = mf(p)
        #Ecoul = (p**2).sum(dim=-1)
        dipole_atom = p.reshape(n_molecule, n_atom, 3)

        # remove COM
        masses = torch.tensor(mass, dtype=dtype, device=device)[species]
        R0 = (coordinates*masses.unsqueeze(2)).sum(dim=1, keepdim=True)/masses.sum(dim=1).reshape(-1,1,1)
        R = coordinates-R0
        if q0 is not None:
            c = 0.5*e2_over_four_pi_epsilon_0*(f_ij*(q0.unsqueeze(1)*q0.unsqueeze(1)).unsqueeze(-1)).sum(dim=2).sum(dim=1) \
                - (q0.reshape(n_molecule, n_atom, 1)*R*Eext.unsqueeze(1)).sum(dim=2).sum(dim=1, keepdim=True)
            dipole0 = (q0.unsqueeze(2)*R).sum(dim=1)
            quadrupole0 = ((q0.unsqueeze(2)*R).unsqueeze(3)*R.unsqueeze(2)).sum(dim=1)
            Ecoul.add_(c)
        else:
            dipole0 = 0.0
            quadrupole0 = 0.0
        
        quadrupole = (R.unsqueeze(-1)*dipole_atom.unsqueeze(-2) + dipole_atom.unsqueeze(-1)*R.unsqueeze(-2)).sum(dim=1) \
            + quadrupole0
        dipole = dipole_atom.sum(dim=1) + dipole0
        #return vec_1, dis_0 #, dis_0 #, dis_2, L_ij
        #return rhat, K, f_ij, L_ij, A, b, dipole, Ecoul.reshape(-1,1), quadrupole, Eext, dipole_atom
        return dipole, Ecoul.reshape(-1,1), quadrupole, Eext, dipole_atom

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x**2).sum(dim=-1, keepdim=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        x, = ctx.saved_tensors
        return 2.0*grad_output*x
