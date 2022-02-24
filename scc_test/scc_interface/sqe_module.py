import torch
import warnings
from .functions import coul_J, pi, E_h, cutoff
from .eem_module import EEM
import warnings

"""
Unit used here
Length: Angstom
Charge: elementary charge +e
Energy: eV
"""

def conjugate_gradient(A, b, x0=None, max_iter = 10000, eps=1.0e-11):
    # dtype, double eps = 1.0e-11 or smaller
    #        float  eps = 1.0e-3 or smaller
    # A: callable, A(x) ==> Ax
    # b: shape (dim,)
    # A: callalbe, return Ax with shape (dim)
    #    behave as a matrix with shape (dim, dim)
    #TODO preconditioning
    if x0 == None:
        r = -b
        p = b
        x = torch.zeros_like(b)
    else: 
        r = A(x0) - b
        p = -r
        x = x0
    if r.abs().max()==0.0:
        return x
    #rTr = torch.matmul(r.transpose(1,2), r) # r.T * r
    rTr = torch.sum(r * r)

    for i in range(max_iter):
        Ap = A(p)
        alpha = rTr / torch.sum(p * Ap)
        
        x.add_(alpha*p)
        r.add_(alpha*Ap)
        tmp = rTr
        rTr = torch.sum(r * r)
        beta = rTr / tmp 
        p = -r+beta*p
        err = torch.sum(torch.abs(r))
        if err<eps:
            break
        if (i+1) >= max_iter:
            warnings.warn("Conjugate Gradient not converged")
            print("residule:", A(p)-b, 'b: ', b, 'p: ', p)
    return x

class ConjugateGradientSymmetric(torch.autograd.Function):

    @staticmethod
    def forward(ctx, b, f_cut_r, K, Ld, J, Ap, n_real_atom, n_molecule, n_atom, nonblank, pair_first, pair_second, real_atoms):
        """
        Ap : take inputs f_cut_r, K, etc and return a callable function A
        A : callable, A(X) ==> torch.matmul(A, X), implicit shape (n, n)
                      A is symmetric (easier for backward)
        b : shape (n,)
        p : shape (n,)
        retrun p = A^{-1} * b
        i.e. solving linear equation A*p=b with conjugate gradient
        """
        A = Ap(f_cut_r, K, Ld, J, n_real_atom, n_molecule, n_atom, nonblank, pair_first, pair_second)
        p = conjugate_gradient(A, b)
        #raise "Bugs here, not sure how to properly save A, as it is a function, and can't just save as class attribute"
        ctx.A = A
        ctx.Ap = Ap
        ctx.n_real_atom = n_real_atom
        ctx.n_molecule = n_molecule
        ctx.n_atom = n_atom
        ctx.save_for_backward(p, b, f_cut_r, K, Ld, J, nonblank, pair_first, pair_second, real_atoms)
        return p
    
    @staticmethod
    def backward(ctx, dp):
        """
        db = A^{-T} * dp
        dA = -A^{-T} dC * C^T = - dB * p^T
        A^T db =  dp, A^T = A, A db = dp
        """
        A = ctx.A
        Ap = ctx.Ap
        n_real_atom = ctx.n_real_atom
        n_molecule = ctx.n_molecule
        n_atom = ctx.n_atom
        p, b, f_cut_r, K, Ld, J, nonblank, pair_first, pair_second, real_atoms = ctx.saved_tensors
        #db = conjugate_gradient(A, dp)
        
        db = ConjugateGradientSymmetric().apply(dp, f_cut_r, K, Ld, J, Ap, n_real_atom, n_molecule, n_atom, nonblank, pair_first, pair_second, real_atoms)
        dtype = p.dtype
        device = p.device

        #dA = - torch.matmul(db, p.T)
        # d (K*f_cut_r**2) = diagonal (dA)
        #dA_diag = -db*p
        #dK = dA_diag*f_cut_r**2
        #df_cut_r = 2.0*dA_diag*K*f_cut_r
        dK = -db*p
        df_cut_r = torch.zeros_like(f_cut_r)

        # A = K + S^T (L+J) S, first part is done above
        # A is symmetric but dA is not
        # d(S^T) = dA S^T *(L+J)^T ==> dS = (L+J) S dA^T
        # d(L+J) = S dA S^T
        # dS = (L+J) S dA 
        # combine d(S^T) and dS ==> dS =  (L+J) S (dA+dA^T)
        # SdA = S * ( - db p^T) = -S*db*p^T
        # SdA^T = S* ( - p db^T) = -S*p * db^T

        Sdb = torch.zeros(n_real_atom, dtype=dtype, device=device)
        Sdb.index_add_(0, pair_second, db*f_cut_r)
        Sdb.index_add_(0, pair_first, -db*f_cut_r)
        LSdb = Ld * Sdb  # diagonal part, shape (n_real_atom, )
        Sdb0 = torch.zeros(n_molecule, n_atom, dtype=dtype, device=device)
        Sdb0[nonblank] = Sdb
        JSdb = torch.matmul(J, Sdb0.unsqueeze(2)).reshape(n_molecule, n_atom)[nonblank] # (n_real_atom, )
        LJSdb = LSdb + JSdb

        pc = p*f_cut_r
        Sp = torch.zeros(n_real_atom, dtype=dtype, device=device)
        Sp.index_add_(0, pair_second, pc)
        Sp.index_add_(0, pair_first, -pc)
        Sp0 = torch.zeros(n_molecule, n_atom, dtype=dtype, device=device)
        Sp0[nonblank] = Sp
        LSp = Ld * Sp  # diagonal part, shape (n_real_atom, )

        Sp0 = torch.zeros(n_molecule, n_atom, dtype=dtype, device=pc.device)
        Sp0[nonblank] = Sp
        JSp = torch.matmul(J, Sp0.unsqueeze(2)).reshape(n_molecule, n_atom)[nonblank] # (n_real_atom, )
        LJSp = LSp + JSp

        # S = f_cut_r * (+1)  (pair_first, pair_second), 
        #   = f_cut_r * (-1)  (pair_second, pair_first), 
        # double counted pairs are removed
        # S =  signS * Diag(f_cut_r)
        # S: shape (n_atom, n_pair), map pairs to atoms
        # signS: + 1 if atom = pair_second
        #        - 1 if atom = pair_first
        # signS has the same shape as S
        # f_cut_r behave as a diagonal matrix with shape (n_pair, n_pair)
        # df_cut_r = signS^T dS = - signS^T (L+J)S (db p^T + p db^T), take diagonal elements
   
        signS_LJSdb = (LJSdb[pair_second] - LJSdb[pair_first])
        df_cut_r.add_(-signS_LJSdb*p)
        signS_LJSp = (LJSp[pair_second] - LJSp[pair_first])
        df_cut_r.add_(-signS_LJSp*db)
        
        #d(L+J) = S dA S^T = S (- db p^T) S^T = -(S db) (S p)^T
        
        # J_ii = 0, diagonal elements of J are 0
        # diagonal elements of L are Ld
        dLd = - Sdb*Sp
        dJ = - torch.matmul(Sdb0.unsqueeze(2), Sp0.unsqueeze(1))

        return db, df_cut_r, dK, dLd, dJ, None, None, None, None, None, None, None, None

class SQE(EEM):

    def __init__(self, f_cutoff='cos', units={'energy':'eV', 'length':"Angstrom"}, lower_bound=0.0):
        """
        Compute Coulomb energy based on atomic charge, position, and external field
        E_{coul} = \frac{1}{2} \sum_{i,j} q_i J_{ij} q_j + \sum_i \phi_{ext,i} q_i
        f_cutoff : callable function, take length input, output scaling
        """
        super().__init__(units=units, lower_bound=lower_bound)
        if callable(f_cutoff):
            self.f_cutoff = f_cutoff
        else:
            self.f_cutoff = cutoff(f_cutoff)

    def forward(self, species, coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, Ld, chi, Eext=None, q0=None, cells=None ):
        """
        species : shape (n_molecule, n_atom)
        coordinates : shape (n_molecule, n_atom, 3), unit Angstrom
        sigma : width, shape (n_molecule, n_atom), unit Angstrom
        K : shape (n_pair, 1), > 0.0, from chemnn.layers.targets.HBondSymmetric
        L : terms like J, presented in E_{corr}, but with shape (n_pair, 1), from chemnn.layers.targets.HBondSymmetric
        Ld : diagonal part of L, as in HIPNN, there is no pair between same atom. shape (n_molecule, n_atom)
        chi : presented in E_{corr}, shape (n_molecule, n_atom), from chemnn.layers.targets.HCharge (with n_target=1)
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
        chi = chi.reshape(-1)

        # make sure sigma is positive and is around 0.5
        dtype = coordinates.dtype
        device = coordinates.device
        n_molecule, n_atom = coordinates.shape[:2]
        nonblank = species>0

        sigma0 = torch.zeros(species.shape, dtype=dtype, device=device)
        sigma0[nonblank] = sigma.reshape(-1)
        sigma = self.sp_L(sigma0)
        
        J = coul_J(nonblank, coordinates, sigma, cells)

        cond = pair_first < pair_second
        pair_first = pair_first[cond]
        pair_second = pair_second[cond]

        #K = self.sp_E(K) + 0.1 #+self.c
        K = self.sp_E(K.reshape(-1)[cond]) + self.c
        Ld = self.sp_E(Ld.reshape(-1)) + self.c

        # phi : potential for atom i, shape (n_molecule, n_atom)
        if Eext == None:
            Eext = torch.zeros(n_molecule, 3, dtype=dtype, device=device, requires_grad=True)
        else:
            assert Eext.dim() == 2, "Eext is not implemented, and not used"
            # Eext is a vector for each molecule, shape (n_molecule, 3)
            Eext.requires_grad_(True)

        phi = -torch.sum(Eext.unsqueeze(1)*coordinates, dim=2)
        
        f_cut_r = self.f_cutoff(pair_dist[cond])

        # b = - S^T ((L+J) q^0 + (chi+phi))
        # Lij : Ld: diagonal part
        # chi : (n_real_atom, )
        # phi[nonblank] : (n_real_atom, )

        if torch.is_tensor(q0):
            q0r = q0[nonblank] # (n_real_atom, )
            Lq0 = Ld * q0r  # diagonal part, shape (n_real_atom, )
            Jq0 = torch.matmul(J, q0.unsqueeze(2)).reshape(n_molecule, n_atom)[nonblank] # (n_real_atom, )
        else:
            q0r = 0.0
            Lq0 = 0.0
            Jq0 = 0.0

        if torch.is_tensor(phi):    
            phir = phi[nonblank] 
        else:
            phir = 0.0
        #print(K.shape, chi.shape, sigma.shape, L.shape, Ld.shape)
        b0 = (Lq0 + Jq0 + chi + phir).reshape(-1)
        # b = -S^T b0      
        b = - (b0[pair_second]-b0[pair_first])*f_cut_r

        # c = 0.5 * q^0^T (L+J) q^0 + q^0^T (chi+ phi)
        c = 0.5*q0r*(Lq0+Jq0) + q0r*(chi + phir)
        
        n_real_atom = real_atoms.shape[0]
        atom_mol_id = real_atoms // n_atom
        pair_mol_id = real_atoms[pair_first] // n_atom

        #p = torch.zeros_like(b)
        def Ap(f_cut_r, K, Ld, J, n_real_atom, n_molecule, n_atom, nonblank, pair_first, pair_second):
            def A(p):
                # A*p = S^T K S p + S^T (L+J) S p
                # S^T K S p =  K S^T S p 
                Ap_1 = K*p #*(f_cut_r**2)
                
                pc = p*f_cut_r
                Sp = torch.zeros(n_real_atom, dtype=dtype, device=pc.device)
                Sp.index_add_(0, pair_second, pc)
                Sp.index_add_(0, pair_first, -pc)

                LSp = Ld * Sp  # diagonal part, shape (n_real_atom, )
                #"""

                Sp0 = torch.zeros(n_molecule, n_atom, dtype=dtype, device=pc.device)
                Sp0[nonblank] = Sp

                JSp = torch.matmul(J, Sp0.unsqueeze(2)).reshape(n_molecule, n_atom)[nonblank] # (n_real_atom, )
                #"""
                #JSp = 0.0
                LJSp = LSp + JSp
                # S^T (L+J)Sp
                Ap_2 = (LJSp[pair_second] - LJSp[pair_first])*f_cut_r
                
                return Ap_1 + Ap_2 
            return A
        
        cg = ConjugateGradientSymmetric.apply

        p = cg(b, f_cut_r, K, Ld, J, Ap, n_real_atom, n_molecule, n_atom, nonblank, pair_first, pair_second, real_atoms)

        # Etot =  0.5 p^T A p - p^T b + c
        # Ap = b
        # Etot = -0.5 p^T b + c
        Etot = torch.zeros(n_molecule, dtype=dtype, device=device)
        Etot.index_add_(0, atom_mol_id, c)
        Etot.index_add_(0, pair_mol_id, -0.5*p*b)
        Sp = torch.zeros(n_real_atom, dtype=dtype, device=device)
        pc = p*f_cut_r

        Sp.index_add_(0, pair_second, pc)
        Sp.index_add_(0, pair_first, -pc)
        if torch.is_tensor(q0):
            q = q0.clone()
        else:
            q = torch.zeros(n_molecule*n_atom, dtype=dtype, device=device)
        
        q.index_add_(0, real_atoms, Sp)
        q = q.reshape(n_molecule, n_atom)
        d = self.dipole(q.unsqueeze(2), coordinates)
        quadrupole = self.quadrupole(q.unsqueeze(2), species, coordinates)

        """
        polarizability
        alpha_{\mu \nu} = - r_{\mu}^T * S A^{-1} S^T * r_{\nu}
        """
        R = coordinates.reshape(-1,3)[nonblank.reshape(-1)]
        ST_R = (R[pair_second,:] - R[pair_first,:])*f_cut_r.unsqueeze(1)

        A_inv_ST_R = torch.zeros_like(ST_R)
        for i in range(3):
            #cg = ConjugateGradientSymmetric.apply
            A_inv_ST_R[:,i] = cg(ST_R[:,i], f_cut_r, K, Ld, J, Ap, n_real_atom, n_molecule, n_atom, nonblank, pair_first, pair_second, real_atoms)

        # #alpha = -torch.matmul(ST_R.transpose(1,2), A_inv_ST_R)
        # alpha0 = torch.zeros(n_molecule,3,3, dtype=dtype, device=device)
        # for i in range(3):
        #     for j in range(i+1):
        #         alpha0[...,i,j].index_add_(0,
        #             pair_mol_id,
        #             -ST_R[:,i]*A_inv_ST_R[:,j]
        #             )
        # alpha = alpha0 + alpha0.tril(-1).transpose(1,2).contiguous()

        return q, Etot.reshape(n_molecule,1), d, quadrupole, Eext #, alpha  
        
if __name__ == "__main__":
    """
    # create testing example
    # two molecules with padding
    """
    #-----------------------------------------------------------------------------------------------------------------
    torch.random.manual_seed(0)
    species = torch.tensor([ [8,1,1,0], [8,1,0,0]], dtype=torch.int64)
    coordinates = torch.tensor([
                                [[0.0, 0.0, 0.0], [0.0, 0.75695, 0.58588], [0.0, -0.75695, 0.58588], [0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.75695, 0.58588], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                            ],dtype=torch.double, requires_grad=True)
    real_atoms = torch.tensor([0,1,2,4,5], dtype=torch.int64)
    pair_first  = torch.tensor([0,0,1,2,3,4], dtype=torch.int64)
    pair_second = torch.tensor([1,2,0,0,4,3], dtype=torch.int64)
    pair_dist = torch.norm(coordinates[0,1].detach())*torch.ones(pair_first.shape, dtype=torch.double)
    pair_dist.requires_grad_(True)
    n_pair = sum(pair_first<pair_second) 
    L = torch.rand(n_pair ,dtype=torch.double, requires_grad=True)
    K = torch.rand(n_pair, dtype=torch.double, requires_grad=True)
    Ld = torch.rand(real_atoms.shape, dtype=torch.double, requires_grad=True)
    chi = torch.rand(real_atoms.shape, dtype=torch.double, requires_grad=True)
    q0 = torch.zeros(species.shape, dtype=torch.double) # q0: bare charge, given outside, not learned, for charged system, if None, treat as 0.0

    sigma = torch.randn(real_atoms.shape, dtype=torch.double)*0.1+0.5
    sigma.requires_grad_(True)
    cells =  None
    Eext = torch.tensor([0.0,0.0,0.0], dtype=torch.double)
    #-----------------------------------------------------------------------------------------------------------------
    sqe = SQE()
    """
    with torch.autograd.set_detect_anomaly(True):
        q, Etot, d, alpha = sqe(species=species, 
                 coordinates=coordinates, 
                 sigma=sigma, 
                 real_atoms=real_atoms, 
                 pair_first=pair_first, 
                 pair_second=pair_second, 
                 pair_dist=pair_dist, 
                 K=K,
                 L=L, 
                 Ld=Ld,
                 chi=chi)
        print(alpha)
        #((q**2).sum()+(Etot**3).sum()).backward() #(create_graph=True, retain_graph=True)
        #print(coordinates.grad, sigma.grad, pair_dist.grad, K.grad, L.grad, Ld.grad, chi.grad, sep="\n")
    #"""
    def fc(species,coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, L, Ld, chi):
        return sqe(species,coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, L, Ld, chi)[:4]
    print(torch.autograd.gradcheck(fc, (species,coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, L, Ld, chi)))
    #print(torch.autograd.gradcheck(fc, (species,coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, L, Ld, chi)))
    print(torch.autograd.gradgradcheck(fc, (species,coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, L, Ld, chi)))#,allow_unused=True))

