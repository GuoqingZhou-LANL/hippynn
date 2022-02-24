import torch
from .functions import cutoff, e2_over_four_pi_epsilon_0, mass, pi

# implement the trunctated smoothly shifted and damped charge and dipole interaction

# For the DEM model, here will provide dipole interaction matrix 
# and charge-dipole interaction vector, charge interaction energy


class DEM_coul(torch.nn.Module):

    def __init__(self, coul_cutoff = 15.0, alpha = 0.2):
        super().__init__()
        self.coul_cutoff = coul_cutoff

        # dammping terms, unit 1/Angstrom, suggest 0.2 - 0.25
        self.alpha = alpha
    

    def forward(self, nonblank, real_atoms, mol_index, sigma, pair_dist, pair_first, pair_second, pair_coord, q0=None):

        device = pair_dist.device
        dtype = pair_dist.dtype

        n_mol, n_atom = nonblank.shape

        rc = torch.tensor(self.coul_cutoff, dtype=dtype, device=device)

        sigma2 = (sigma**2).reshape(-1)
        #ss = sigma2[pair_first] + sigma2[pair_second]
        #ss_sqrt = torch.sqrt(ss)

        pair_dist2 = pair_dist ** 2

        f_ij = torch.erf(pair_dist / torch.sqrt( sigma2[pair_first] + sigma2[pair_second]))
        #f_ij = torch.erf( pair_dist / ss_sqrt)
        #d_f_ij = 2.0/pi**0.5 * torch.exp( -pair_dist2/ss ) / ss_sqrt

        #f_ij_rc = torch.erf( rc/ss_sqrt)
        #d_f_ij_rc = 2.0/pi**0.5 * torch.exp( -rc**2/ss ) / ss_sqrt

        damp = torch.erfc(self.alpha * pair_dist)
        #d_damp = -2.0/pi**0.5 * torch.exp(- self.alpha**2 * pair_dist ** 2) * self.alpha
        d_damp = -2.0/pi**0.5 * torch.exp(- self.alpha**2 * pair_dist2) * self.alpha

        damp_rc = torch.erfc(self.alpha * rc)
        d_damp_rc = -2.0/pi**0.5 * torch.exp(- rc**2 * pair_dist2) * self.alpha

        c0 = pair_dist > self.coul_cutoff
        
        if q0 is None:
            Eqq = 0.0
            bq = 0.0
        else:
            q = q0[nonblank]
            # Uqqs0 = f_ij * ( 1.0/pair_dist - 1.0/rc + (pair_dist - rc)/rc**2 )
            Uqqs0 = f_ij*( damp/pair_dist - damp_rc/rc - (pair_dist - rc)*(d_damp_rc/rc - damp_rc/rc**2 ) )
            # Uqqs0 = f_ij*damp/pair_dist - f_ij_rc * damp_rc / rc - (pair_dist - rc)* ( d_f_ij_rc * damp_rc/rc + f_ij_rc*d_damp_rc/rc - f_ij_rc*damp_rc/rc**2)
            Uqqs = torch.where(c0, 0.0, Uqqs0)
            #Uqqs[c0] = 0.0
            Eqq0 = torch.zeros(n_mol, dtype=dtype, device=device)
            Eqq0.index_add_(0, mol_index[pair_first], 
                q[pair_first] * q[pair_second] * Uqqs)
            Eqq = 0.5*e2_over_four_pi_epsilon_0 * Eqq0.unsqueeze(1)
            #print(Eqq)

            # Uqps0 = f_ij* (1.0/pair_dist2 - 1.0/rc**2 + 2.0*(pair_dist-rc)/rc**3 )
            Uqps0 = f_ij * ( damp/pair_dist2 - damp_rc/rc**2 - (pair_dist - rc)*( d_damp_rc/rc**2 - 2.0*damp_rc/rc**3 ) )
            # Uqps0 = f_ij*damp/pair_dist2 - f_ij_rc*damp_rc/rc**2 - (pair_dist - rc)* ( d_f_ij_rc * damp_rc /rc**2 + f_ij_rc*d_damp_rc/rc**2 - 2.0*f_ij_rc*damp_rc/rc**3 )
            #Uqps[c0] = 0.0
            Uqps = torch.where(c0, 0.0, Uqps0)
            bq0 = (q[pair_second] * Uqps/pair_dist).unsqueeze(1) * pair_coord
            bq1 = torch.zeros(nonblank.sum(), 3, dtype=dtype, device=device)
            bq = torch.zeros(n_mol, n_atom, 3, dtype=dtype, device=device)

            bq1.index_add_(0, pair_first, bq0)
            bq[nonblank] = bq1*e2_over_four_pi_epsilon_0
            bq = bq.reshape(n_mol, -1)

        #print(Eqq)
        #Upps0 = f_ij*( 1.0/pair_dist**3 - 1.0/rc**3 + 3.0*(pair_dist-rc)/rc**4 )
        Upps0 = f_ij * ( damp/pair_dist**3 - damp_rc/rc**3 - (pair_dist - rc)* (d_damp_rc/rc**3 - 3.0*damp_rc/rc**4) )
        # Upps0 = f_ij*damp/pair_dist**3 - f_ij_rc*damp_rc/rc**3 - (pair_dist -rc)* ( d_f_ij_rc*damp_rc/rc**3 + f_ij_rc*d_damp_rc/rc**3 - 3.0*f_ij_rc*damp_rc/rc**4)

        Upps = torch.where(c0, 0.0, Upps0)
        App0 = (e2_over_four_pi_epsilon_0 * Upps ).reshape(-1,1,1) * \
            (
                3.0*pair_coord.unsqueeze(1)*pair_coord.unsqueeze(2)/(pair_dist**2).reshape(-1,1,1) - \
                torch.eye(3, dtype=dtype, device=device).reshape(1,3,3) \
            )
        pair_mask = real_atoms[pair_first]*n_atom+real_atoms[pair_second].remainder(n_atom)
        App1 = torch.zeros(n_mol*n_atom*n_atom, 3, 3, dtype=dtype, device=device)
        App1.index_add_(0, pair_mask, App0)
        App = App1.reshape(n_mol, n_atom, n_atom, 3, 3).permute(0,1,3,2,4).reshape(n_mol, n_atom*3, n_atom*3)

        return App, bq, Eqq

from .dem_module import DEM

class DEM_PBC(DEM):
    
    def __init__(self, coul_cutoff = 15.0, alpha = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dem_coul = DEM_coul(coul_cutoff=coul_cutoff, alpha=alpha)
        self.coul_cutoff = coul_cutoff
        self.alpha = alpha

    def forward(self, species, coordinates, pair_first, pair_coord, w0, w1, real_atoms, mol_index, sigma, pair_dist1, pair_first1, pair_second1, pair_coord1, q0=None, Eext=None ):
        """
        pair_first, pair_coord : computed with short cutoff, used in normal HIPNN
        pair_dist1, pair_first1, pair_second1, pair_coord1, mol_index : computed with a large cutoff ( ~15 Angstrom) and used for truncated coulomb interactions
        """
        n_molecule, n_atom = species.shape
        dtype = pair_coord.dtype
        device = pair_coord.device
        nonblank = species > 0
        n_real_atom = nonblank.sum()

        App, bq, Eqq = self.dem_coul(nonblank, real_atoms, mol_index, sigma, pair_dist1, pair_first1, pair_second1, pair_coord1, q0)

        rhat = pair_coord/torch.linalg.norm(pair_coord, dim=-1, keepdim=True)
        chi = torch.zeros((n_real_atom, 3), dtype=dtype, device=device)
        chi.index_add_(0,pair_first,w0*rhat)

        K = torch.zeros((n_real_atom, 3, 3), dtype=dtype, device=device)
        rt = rhat.unsqueeze(1)*rhat.unsqueeze(2)
        K.index_add_(0,pair_first, w1.unsqueeze(2)*rt)

        chi0 = torch.zeros((n_molecule, n_atom, 3), dtype=dtype, device=device)
        chi0[nonblank] = chi
        chi = chi0

        K0 =  torch.zeros((n_molecule, n_atom, 3, 3), dtype=dtype, device=device)
        K0[nonblank] = K
        K = K0

        # phi : potential for atom i, shape (n_molecule, n_atom)
        if Eext == None:
            Eext = torch.zeros(n_molecule, 3, dtype=dtype, device=device, requires_grad=True)
        else:
            assert Eext.dim() == 2, "Eext is not implemented, and not used"
            # Eext is a vector for each molecule, shape (n_molecule, 3)
            Eext.requires_grad_(True)
        
        #print((App-App.permute((0,2,1))).abs().max())
        #tmp = self.c * torch.eye(n_atom*3, dtype=dtype, device=device ).unsqueeze(0)
        #tmp = torch.diag_embed(K.permute(0,2,3,1)).permute((0,3,1,4,2)).reshape(n_molecule, n_atom*3, n_atom*3)
        #print((tmp-tmp.permute((0,2,1))).abs().max())
        
        A = e2_over_four_pi_epsilon_0 * \
            ( torch.diag_embed(K.permute(0,2,3,1)).permute((0,3,1,4,2)).reshape(n_molecule, n_atom*3, n_atom*3) 
            + self.c * torch.eye(n_atom*3, dtype=dtype, device=device ).unsqueeze(0) ) + App
        
        #
        
        b = (chi + Eext.unsqueeze(1)).reshape(n_molecule, -1) + bq

        p = torch.linalg.solve(A, b)
        Ecoul = 0.5*torch.matmul(p.unsqueeze(1), torch.matmul(A, p.unsqueeze(2))) - torch.matmul(b.unsqueeze(1), p.unsqueeze(2))
        dipole_atom = p.reshape(n_molecule, n_atom, 3)

        # remove COM
        masses = torch.tensor(mass, dtype=dtype, device=device)[species]
        R0 = (coordinates*masses.unsqueeze(2)).sum(dim=1, keepdim=True)/masses.sum(dim=1).reshape(-1,1,1)
        R = coordinates-R0
        if q0 is not None:
            c = Eqq \
                - (q0.reshape(n_molecule, n_atom, 1)*R*Eext.unsqueeze(1)).sum(dim=2).sum(dim=1, keepdim=True)
            dipole0 = (q0.unsqueeze(2)*R).sum(dim=1)
            quadrupole0 = ((q0.unsqueeze(2)*R).unsqueeze(3)*R.unsqueeze(2)).sum(dim=1)
            Ecoul.add_(c.unsqueeze(2))
        else:
            dipole0 = 0.0
            quadrupole0 = 0.0
        
        quadrupole = (R.unsqueeze(-1)*dipole_atom.unsqueeze(-2) + dipole_atom.unsqueeze(-1)*R.unsqueeze(-2)).sum(dim=1) \
            + quadrupole0
        dipole = dipole_atom.sum(dim=1) + dipole0
        
        
        return dipole, Ecoul.reshape(-1,1), quadrupole, Eext, dipole_atom
