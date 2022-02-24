import torch
import numpy as np

pi = np.pi

# E_h = e^2/(4 \pi \epsilon_0 a_0)
# e^2 / (4 \pi \epsilon_0) = E_h * a_0
a_0 =  0.529177210903 # Bohr radius in Angstrom
E_h = 27.211386245988 # Hatree energy in eV
e2_over_four_pi_epsilon_0 = E_h*a_0

#TODO with Fast Ewald Summation PPPM
#TODO or use screened coulomb

def coul_J(nonblank, coordinates, sigma, cells=None):

    n_molecule, n_atom, _ = coordinates.shape
    device = coordinates.device
    dtype = coordinates.dtype
    if cells != None:
        raise ValueError("Coulomb is not implemented for periodic systems yet")

    one = torch.tensor(1.0,dtype=dtype,device=device)
    zero = torch.zeros_like(one)
    #rij0 =  torch.linalg.norm(coordinates.unsqueeze(1)-coordinates.unsqueeze(2),dim=-1)
    rij0 = torch.norm(coordinates.unsqueeze(1)-coordinates.unsqueeze(2),dim=-1)
    mask = nonblank.unsqueeze(1)*nonblank.unsqueeze(2)
    rij1 = torch.where(mask, rij0, one)
    # diagonal 
    maskd = torch.eye(n_atom,device=device,dtype=torch.int).bool().unsqueeze(0)
    rij = torch.where( maskd, one, rij1)
     
    sigma2 = sigma**2
    ss0 = sigma2.unsqueeze(1)+sigma2.unsqueeze(2)
    ss = torch.where(mask, ss0, one)
    
    J0 = ( e2_over_four_pi_epsilon_0 / rij ) * torch.erf( rij / torch.sqrt( ss ) )

    # 1. \delta_{i,j}, remove i==j
    # 2. padding, put padding part as 0

    J1 = torch.where(mask, J0, zero)
    J = torch.where(maskd, zero, J1)

    return J

def trim(x):
    with torch.no_grad():
        if torch.is_tensor(x):
            x[~torch.isfinite(x)] = 0.0
    return x

def check(x):
    if torch.is_tensor(x):
        if (~torch.isfinite(x)).any():
            print(x)
            raise ValueError("nan/inf in x")

class cutoff(torch.nn.Module):

    def __init__(self, type='cos', cutoff=4.0):
        """
        cutoff function collections
        """
        super().__init__()
        self.type = type
        self.cutoff = cutoff
    
    def cos(self,x):
        y = 0.5*torch.cos( (x/self.cutoff)*pi ) + 0.5
        y[x>self.cutoff] = 0.0
        return y
    
    def forward(self,x):
        if self.type == 'cos':
            return self.cos(x)
        else:
            raise ValueError(str(self.type) + " cutoff function not impletemented yet")

index = {
"X"  :0  ,"H"  :1  ,"HE" :2  ,"LI" :3  ,"BE" :4  ,"B"  :5  ,"C"  :6  ,
"N"  :7  ,"O"  :8  ,"F"  :9  ,"NE" :10 ,"NA" :11 ,"MG" :12 ,"AL" :13 ,
"SI" :14 ,"P"  :15 ,"S"  :16 ,"CL" :17 ,"AR" :18 ,"K"  :19 ,"CA" :20 ,
"SC" :21 ,"TI" :22 ,"V"  :23 ,"CR" :24 ,"MN" :25 ,"FE" :26 ,"CO" :27 ,
"NI" :28 ,"CU" :29 ,"ZN" :30 ,"GA" :31 ,"GE" :32 ,"AS" :33 ,"SE" :34 ,
"BR" :35 ,"KR" :36 ,"RB" :37 ,"SR" :38 ,"Y"  :39 ,"ZR" :40 ,"NB" :41 ,
"MO" :42 ,"TC" :43 ,"RU" :44 ,"RH" :45 ,"PD" :46 ,"AG" :47 ,"CD" :48 ,
"IN" :49 ,"SN" :50 ,"SB" :51 ,"TE" :52 ,"I"  :53 ,"XE" :54 ,"CS" :55 ,
"BA" :56 ,"LA" :57 ,"CE" :58 ,"PR" :59 ,"ND" :60 ,"PM" :61 ,"SM" :62 ,
"EU" :63 ,"GD" :64 ,"TB" :65 ,"DY" :66 ,"HO" :67 ,"ER" :68 ,"TM" :69 ,
"YB" :70 ,"LU" :71 ,"HF" :72 ,"TA" :73 ,"W"  :74 ,"RE" :75 ,"OS" :76 ,
"IR" :77 ,"PT" :78 ,"AU" :79 ,"HG" :80 ,"TL" :81 ,"PB" :82 ,"BI" :83 ,
"PO" :84 ,"AT" :85 ,"RN" :86 ,"FR" :87 ,"RA" :88 ,"AC" :89 ,"TH" :90 ,
"PA" :91 ,"U"  :92 ,"NP" :93 ,"PU" :94 ,"AM" :95 ,"CM" :96 ,"BK" :97 ,
"CF" :98 ,"ES" :99 ,"FM" :100,"MD" :101,"NO" :102,"LR" :103,"RF" :104,
"DB" :105,"SG" :106,"BH" :107,"HS" :108,"MT" :109,"DS" :110,"RG" :111,
"UUB":112,"UUT":113,"UUQ":114,"UUP":115,"UUH":116,"UUS":117,"UUO":118 }

mass = np.asarray([
0.0,1.00782503207,4.00260325415,7.016004548,9.012182201,11.009305406,
12,14.00307400478,15.99491461956,18.998403224,19.99244017542,
22.98976928087,23.985041699,26.981538627,27.97692653246,30.973761629,
31.972070999,34.968852682,39.96238312251,38.963706679,39.962590983,
44.955911909,47.947946281,50.943959507,51.940507472,54.938045141,
55.934937475,58.933195048,57.935342907,62.929597474,63.929142222,
68.925573587,73.921177767,74.921596478,79.916521271,78.918337087,
85.910610729,84.911789737,87.905612124,88.905848295,89.904704416,
92.906378058,97.905408169,98.906254747,101.904349312,102.905504292,
105.903485715,106.90509682,113.90335854,114.903878484,119.902194676,
120.903815686,129.906224399,126.904472681,131.904153457,132.905451932,
137.905247237,138.906353267,139.905438706,140.907652769,141.907723297,
144.912749023,151.919732425,152.921230339,157.924103912,158.925346757,
163.929174751,164.93032207,165.930293061,168.93421325,173.938862089,
174.940771819,179.946549953,180.947995763,183.950931188,186.955753109,
191.96148069,192.96292643,194.964791134,196.966568662,201.970643011,
204.974427541,207.976652071,208.980398734,208.982430435,210.987496271,
222.017577738,222.01755173,228.031070292,227.027752127,232.038055325,
231.03588399,238.050788247,237.048173444,242.058742611,243.06138108,
247.07035354,247.07030708,251.079586788,252.082978512,257.095104724,
258.098431319,255.093241131,260.105504,263.112547,255.107398,259.114500,
262.122892,263.128558,265.136151,281.162061,272.153615,283.171792,283.176451,
285.183698,287.191186,292.199786,291.206564,293.214670])

def get_mass(s):
    return mass[index[s.upper()]]

class polarizability(torch.nn.Module):
    
    def forward(self, dipole, Eext):
        alpha = torch.zeros(dipole.shape[0],3,3,dtype=dipole.dtype,device=dipole.device)
        for k in range(3):
            alpha[...,:,k] = torch.autograd.grad(
                dipole[...,k].sum(), Eext, create_graph=True)[0]
        return alpha

class quadrupole(torch.nn.Module):

    def __init__(self, traceless=True):
        super().__init__()
        self.traceless = traceless

    def forward(self, q0,species,coordinates):
        """
        q : charge, shape (n_molecule, n_atom, 1)
        coordinates : shape (n_molecule, n_atom, 3)
        species : shape (n_molecule, n_atom)
        """
        if q0.dim==1:
            q = torch.zeros(species.shape,dtype=q0.dtype, device=q0.device)
            q[species>0]=q0
            q.unsqueeze_(2)
        else:
            q = q0
        # remove center of mass
        masses = torch.tensor(mass, dtype=q.dtype, device=q.device)[species]
        R0 = (coordinates*masses.unsqueeze(2)).sum(dim=1, keepdim=True)/masses.sum(dim=1).reshape(-1,1,1)
        R = coordinates-R0
        quadrupole = ((q*R).unsqueeze(3)*R.unsqueeze(2)).sum(dim=1)
        # traceless form
        if self.traceless:
            d = (quadrupole[...,0,0] + quadrupole[...,1,1] + quadrupole[...,2,2])/3.0
            for k in range(3):
                quadrupole[...,k,k] -= d
        return quadrupole

class SplitMatrix(torch.nn.Module):
    """
    return the diagonal and offdiagonal elements
    remove trace if needed
    """
    def __init__(self, traceless=False):
        super().__init__()
        self.traceless = traceless
    
    def forward(self, x):
        diag = torch.cat([x[...,0:1,0], x[...,1:2,1], x[...,2:3,2]], dim=-1)
        offdiag = torch.cat([x[...,0:1,1], x[...,1:2,2], x[...,2:3,0]], dim=-1)
        if self.traceless:
            diag1 = diag - diag.mean(dim=-1, keepdim=True)
            return diag1, offdiag
        else:
            return diag, offdiag

