import numpy as np
import os

au = 51.422062351130926 # electric field in a.u. 1.0 Eh/Bohr/e = 51.422062351130926 eV/Angstrom/e

nmax=6
N=9742
Eh = 27.2113834 # 1 Hartree = 27.2113834 eV
Bohr = 0.5291772083 # 1 Bohr = 0.5291772083 Angstrom

# Etot: total energy (scf energy + vdw correction), unit: eV
Etot = np.zeros(N, dtype=np.float64)
# Escf: scf energy (total energy without D3 correction), unit: eV
Escf = np.zeros(N, dtype=np.float64)
# Grad: gradient ( -1.0 * force), unit: eV/Angstrom
Grad = np.zeros((N,6,3), dtype=np.float64)
# MULLIKEN ATOMIC CHARGES, unit +e (i.e. proton will have +1)
Mulliken = np.zeros((N,6), dtype=np.float64)
# LOEWDIN ATOMIC CHARGES, unit +e (i.e. proton will have +1)
Loewdin = np.zeros((N,6), dtype=np.float64)

nprop = 3  # how many different type of properties are output in mol_property.txt
# dipole moment, unit +e*Angstrom
dipole = np.zeros((N,3,nprop), dtype=np.float64)
# quadrupole moment, unit +e*Angstrom^2
quadrupole = np.zeros((N,3,3,nprop), dtype=np.float64)
# polarizability, unit e^2*Angstrom^2/eV
polarizability = np.zeros((N,3,3,2), dtype=np.float64)

for c in range(N):
    fdir = "./files/%d" % c
    if os.path.exists(fdir) and os.path.exists(fdir+"/log") and ("TOTAL RUN TIME: 0 days" in os.popen("tail -n 1 ./files/%d/log " % c).read().strip()):
        print(c)
        natom = 6 #int(os.popen("head -n 1 ./%s/input.xyz" % fdir).read().strip().split()[0])

        # change unit for energy from atomic unit to eV
        Etot[c] = float(os.popen("grep energy %s/mol.engrad -A 2 | tail -n 1" % fdir).read().strip().split()[0])*Eh

        # change unit for energy from atomic unit to eV
        Escf[c] = float(os.popen("grep 'SCF Energy:' %s/mol_property.txt" % fdir).read().strip().split()[-1])*Eh

        # change unit for gradient from atomic unit to eV/Angstrom
        g = np.zeros((1,nmax,3),dtype=np.float64)
        g[0,:natom] =  np.asarray([float(x.strip().split()[0]) for x in os.popen("grep gradient %s/mol.engrad -A %d | tail -n %d" % (fdir, natom*3+1, natom*3)).read().split("\n")[:-1]]).reshape(-1,3)*Eh/Bohr
        Grad[c] = g[0]

        # MULLIKEN charge, unit +e 
        c1 = np.zeros((1,nmax),dtype=np.float64)
        c1[0,:natom] = np.asarray( \
            [ float(x.strip().split()[-1]) \
                for x in os.popen("grep 'MULLIKEN ATOMIC CHARGES' %s/log -A %d | tail -n %d" % (fdir, natom+1, natom)).read().split("\n")[:-1] \
            ] \
            )
        Mulliken[c] = c1

        # LOEWDIN charge, unit +e 
        c2 = np.zeros((1,nmax),dtype=np.float64)
        c2[0,:natom] = np.asarray( \
            [ float(x.strip().split()[-1]) \
                for x in os.popen("grep 'LOEWDIN ATOMIC CHARGES' %s/log -A %d | tail -n %d" % (fdir, natom+1, natom)).read().split("\n")[:-1] \
            ] \
            )
        Loewdin[c] = c2

        # Change unit for dipole from atomic unit (e*Bohr) to e*Angstrom
        tmp = os.popen("grep 'Total Dipole moment:' %s/mol_property.txt -A 4" % fdir).read().strip().split('\n')
        for k in range(3):
            dipole[c,k,:] = np.asarray( [ float(x.strip().split()[-1]) for x in tmp[(k+2)::6]])*Bohr
        #print(dipole[c,:,0])

        # Change unit for quadrupole from atomic unit (e*Bohr^2) to e*Angstrom^2
        tmp = os.popen("grep 'Total quadrupole moment' %s/mol_property.txt -A 4" % fdir).read().strip().split('\n')
        for k in range(3):
            quadrupole[c,k,:,:] = np.asarray( [[ float(y) for y in x.strip().split()[-3:]] for x in tmp[(k+2)::6]]).T*Bohr**2
        #print(quadrupole[c,:,:,0])

        # polarizability, change unit from atomic unit to e^2*Bohr*2/Eh e^2*Angstrom^2/eV
        tmp = os.popen("grep 'The raw cartesian tensor (atomic units):' %s/mol_property.txt -A 4" % fdir).read().strip().split('\n')
        for k in range(3):
            polarizability[c,k,:,:] = np.asarray( [[ float(y) for y in x.strip().split()[-3:]] for x in tmp[(k+2)::6]]).T*Bohr**2/Eh
        #print(polarizability[c,:,:,0])

np.save("water_dimer_MP2-Etot.npy", Etot)
np.save("water_dimer_MP2-Escf.npy", Escf)
np.save("water_dimer_MP2-Grad.npy", Grad)
np.save("water_dimer_MP2-Mulliken.npy", Mulliken)
np.save("water_dimer_MP2-Loewdin.npy", Loewdin)
np.save("water_dimer_MP2-dipole0.npy", dipole[:,:,0])
np.save("water_dimer_MP2-dipole1.npy", dipole[:,:,1])
np.save("water_dimer_MP2-dipole2.npy", dipole[:,:,2])
np.save("water_dimer_MP2-quadrupole0.npy", quadrupole[:,:,:,0])
np.save("water_dimer_MP2-quadrupole1.npy", quadrupole[:,:,:,1])
np.save("water_dimer_MP2-quadrupole2.npy", quadrupole[:,:,:,2])
np.save("water_dimer_MP2-polarizability0.npy", polarizability[:,:,:,0])
np.save("water_dimer_MP2-polarizability1.npy", polarizability[:,:,:,1])
