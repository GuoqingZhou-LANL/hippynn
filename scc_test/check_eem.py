import torch
from scc_interface.eem_module import EEM

# create testing example
# two molecules with padding
device = torch.device("cpu")
species = torch.tensor([ [8,1,1,0], [8,1,0,0]], dtype=torch.int64, device=device)
coordinates = torch.tensor([
                            [[0.0, 0.0, 0.0], [0.0, 0.75695, 0.58588], [0.0, -0.75695, 0.58588], [0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0], [0.0, 0.75695, 0.58588], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                            ],dtype=torch.double, device=device)
#print(coordinates.shape)
real_atoms = torch.tensor([0,1,2,4,5], dtype=torch.int64, device=device).reshape(-1,1)
Ld = torch.rand(real_atoms.shape,dtype=torch.double, device=device)
chi = torch.rand(real_atoms.shape,dtype=torch.double, device=device)
q_mol = torch.tensor([1.0,-1.0], dtype=torch.double, device=device) # or None or 0.0

nonblank = species>0
sigma = torch.randn(real_atoms.shape, dtype=torch.double, device=device)*0.1+0.5

c = EEM()
coordinates.requires_grad_(True)
sigma.requires_grad_(True)
Ld.requires_grad_(True)
chi.requires_grad_(True)
"""
q, Ecoul, d, alpha = c(species=species, 
                coordinates=coordinates, 
                sigma=sigma, 
                Ld=Ld,
                chi=chi,
                q_mol=q_mol)
print(q, q.sum(dim=1), q_mol, sep="\n")
#Loss = (q**3).sum() + (Ecoul**2).sum()
#Loss.backward()
#Loss = Ecoul.sum()
#Loss = (d**2).sum() + (alpha**2).sum()
#"""
print(torch.autograd.gradcheck(c, (species, coordinates, sigma, Ld, chi), atol=1.0e-7, rtol=1.0e-6))
print(torch.autograd.gradgradcheck(c, (species, coordinates, sigma, Ld, chi), atol=1.0e-5, rtol=1.0e-3))
