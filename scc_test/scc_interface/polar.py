import torch

class polarizability(torch.nn.Module):
    
    def forward(self, dipole, Eext):
        alpha = torch.zeros(dipole.shape[0],3,3,dtype=dipole.dtype,device=dipole.device)
        for k in range(3):
            alpha[...,:,k] = torch.autograd.grad(
                dipole[...,k].sum(), Eext, create_graph=True)[0]
        return alpha