import torch
from scc_interface.sqep_module import SQEP
from scc_interface.sqe_module import SQE
"""
# create testing example
# two molecules with padding
"""
#-----------------------------------------------------------------------------------------------------------------
torch.random.manual_seed(0)
device = torch.device("cpu")
species = torch.tensor([ [8,1,1,0], [8,1,0,0]], dtype=torch.int64, device=device)
coordinates = torch.tensor([
                            [[0.0, 0.0, 0.0], [0.0, 0.75695, 0.58588], [0.0, -0.75695, 0.58588], [0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0], [0.0, 0.75695, 0.58588], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                        ],dtype=torch.double, requires_grad=True, device=device)
real_atoms = torch.tensor([0,1,2,4,5], dtype=torch.int64, device=device)
pair_first  = torch.tensor([0,0,1,2,3,4], dtype=torch.int64, device=device)
pair_second = torch.tensor([1,2,0,0,4,3], dtype=torch.int64, device=device)
pair_dist = torch.norm(coordinates[0,1].detach())*torch.ones(pair_first.shape, dtype=torch.double, device=device)
pair_dist.requires_grad_(True)
n_pair = sum(pair_first<pair_second) 
K = torch.rand(pair_first.shape[0], dtype=torch.double, requires_grad=True, device=device)
Ld = torch.rand(real_atoms.shape, dtype=torch.double, requires_grad=True, device=device).reshape(-1,1)
chi = torch.rand(real_atoms.shape, dtype=torch.double, requires_grad=True, device=device).reshape(-1,1)
q0 = torch.zeros(species.shape, dtype=torch.double, device=device) # q0: bare charge, given outside, not learned, for charged system, if None, treat as 0.0
q0[0,0]=1.0

sigma = torch.randn(real_atoms.shape, dtype=torch.double, device=device).reshape(-1,1)*0.1+0.5
sigma.requires_grad_(True)
Eext = torch.tensor([0.0,0.0,0.0], dtype=torch.double, device=device)
#-----------------------------------------------------------------------------------------------------------------
#sqe = SQEP()
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
                Ld=Ld,
                chi=chi,
                Eext=Eext,
                q0=q0)
    print(q0, q.sum(dim=1), q, d, alpha, sep="\n")
    
    #((q**2).sum()+(Etot**3).sum()).backward()
    #print(coordinates.grad, sigma.grad, pair_dist.grad, K.grad, L.grad, Ld.grad, chi.grad, sep="\n")
#"""
print(torch.autograd.gradcheck(sqe, (species,coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, Ld, chi), atol=1.0e-7, rtol=1.0e-6))
print(torch.autograd.gradgradcheck(sqe, (species,coordinates, sigma, real_atoms, pair_first, pair_second, pair_dist, K, Ld, chi), atol=1.0e-5, rtol=1.0e-3))


