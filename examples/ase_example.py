"""
Script running an aluminum model with ASE.
For training see `ani_aluminum_example.py`.
This will generate the files for a model.

Modified from ase MD example.

If a GPU is available, this script
will use it, and run a somewhat bigger system.
"""

# Imports
import numpy as np
import torch
import hippynn
import ase
import time

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory
from hippynn.interfaces.ase_interface import HippynnCalculator
import ase.build
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

# Load the files
try:
    with active_directory("TEST_ALUMINUM_MODEL", create=False):
        bundle = load_checkpoint_from_cwd(map_location='cpu',restore_db=False)
except FileNotFoundError:
    raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")

model = bundle["training_modules"].model


# Build the calculator
energy_node = model.node_from_name("energy")
calc = HippynnCalculator(energy_node, en_unit=units.eV)
calc.to(torch.float64)
if torch.cuda.is_available():
    nrep = 30  # 27,000 atoms -- should fit in a 12 GB GPU if using custom kernels.
    calc.to(torch.device('cuda'))
else:
    nrep = 10  # 1,000 atoms.

# Build the atoms object
atoms = ase.build.bulk('Al', crystalstructure='fcc', a=4.05)
reps = nrep*np.eye(3, dtype=int)
atoms = ase.build.make_supercell(atoms, reps, wrap=True)
atoms.calc = calc

print("Number of atoms:", len(atoms))

atoms.rattle(.1)
MaxwellBoltzmannDistribution(atoms, temp=500*units.kB)
dyn = VelocityVerlet(atoms, 1*units.fs)


# Simple tracker of the simulation progress, this is not needed to perform MD.
class Tracker():
    def __init__(self, dyn, atoms):
        self.last_call_time = time.time()
        self.last_call_steps = 0
        self.dyn = dyn
        self.atoms = atoms

    def update(self):
        now = time.time()
        diff = now-self.last_call_time
        diff_steps = dyn.nsteps - self.last_call_steps
        try:
            time_per_atom_step = diff/(len(atoms)*diff_steps)
        except ZeroDivisionError:
            time_per_atom_step = float('NaN')
        self.last_call_time = now
        self.last_call_steps = dyn.nsteps
        return time_per_atom_step

    def print(self):
        time_per_atom_step = self.update()
        """Function to print the potential, kinetic and total energy"""
        simtime = round(self.dyn.get_time() / (1000*units.fs), 3)
        print("Simulation time so far:",simtime,"ps")
        print("Performance:",round(1e6*time_per_atom_step,1)," microseconds/(atom-step)")
        epot = self.atoms.get_potential_energy() / len(self.atoms)
        ekin = self.atoms.get_kinetic_energy() / len(self.atoms)
        print('Energy per atom: Epot = %.7feV  Ekin = %.7feV (T=%3.0fK)  '
              'Etot = %.7feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


# Now run the dynamics
tracker = Tracker(dyn, atoms)
tracker.print()
for i in range(100):  # Run 2 ps
    dyn.run(20)  # Run 20 fs
    tracker.print()
