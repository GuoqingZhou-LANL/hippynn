# ---------------- #
# Imported Modules #
# ---------------- #

import os
import sys
import numpy as np
import torch
from scc_interface.scc_nodes import EEMNode, SQEPNode, PolarNode, SQENode, \
    EEMEFNode, SQEEFNode, SQEPEFNode, ExternalFieldNode, MolTensorNode, AlphaENode, AlphaEENode
from hippynn.interfaces.pyseqm_interface.callback import save_and_stop_after
#np.set_printoptions(threshold=np.inf)

#torch.cuda.set_device(0) # Don't try this if you want CPU training!

import matplotlib
matplotlib.use('agg')

import hippynn 

hippynn.custom_kernels.set_custom_kernels (False)

dataset_name = 'water_dimer-'    # Prefix for arrays in folder
dataset_path = os.path.join(os.path.dirname(__file__), "files/data")

netname = 'TEST'
dirname = netname
if not os.path.exists(dirname):
    os.mkdir(dirname)
else:
    pass
    #raise ValueError("Directory {} already exists!".format(dirname))
os.chdir(dirname)

TAG = 0 #int(sys.argv[1])  # False (0): first run, True(n): continue

dtype=torch.float64
torch.set_default_dtype(dtype)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Log the output of python to `training_log.txt`
with hippynn.tools.log_terminal("training_log_tag_%d.txt" % TAG,'wt'):# and torch.autograd.set_detect_anomaly(True):

    # Hyperparameters for the network

    network_params = {
        "possible_species": [0,1,6,7,8],   # Z values of the elements
        'n_features': 40,                     # Number of neurons at each layer
        "n_sensitivities": 20,                # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.7,  # qm7 1.7  qm9 .85  AL100 .85
        "dist_soft_max": 5.,  # qm7 10.  qm9 5.   AL100 5.
        "dist_hard_max": 7.5,  # qm7 15.  qm9 7.5  AL100 7.5
        "n_interaction_layers": 2,            # Number of interaction blocks
        "n_atom_layers": 3,                   # Number of atom layers in an interaction block
    }

    # Define a model

    from hippynn.graphs import inputs, networks, targets, physics

    species = inputs.SpeciesNode(db_name="Z")

    positions = inputs.PositionsNode(db_name="R")

    Eext = ExternalFieldNode(db_name="Eext")

    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs = network_params)

    #henergy = EEMNode("EEM", network)
    #henergy = SQEPNode("SEQP", network)
    #henergy = SQENode("SEQ", network)
    #henergy = EEMEFNode("EEM", (network, Eext))
    #henergy = SQEPEFNode("SEQP", (network, Eext))
    henergy = SQEEFNode("SEQ", (network, Eext))

    network1 = networks.Hipnn("HIPNN", network.parents, module_kwargs = network_params)
    henergy1 = targets.HEnergyNode("HEnergy",network1)

    ef = henergy.external_field
    charge = henergy.charge
    dipole1 = henergy.dipole
    quadru1 = henergy.quadrupole
    polar1 = PolarNode('Polar', (dipole1, ef))

    quadrumol = MolTensorNode("QuadruMol", network)
    quadrupole = quadru1 + quadrumol
    
    polarmol = MolTensorNode("PolarMol", network)
    polarizability = polar1 + polarmol

    dipolemol = AlphaENode('DipoleMole', (polarmol, ef))
    dipole = dipole1 + dipolemol

    Vmol = -0.5*AlphaEENode('Vmol', (polarmol, Eext))

    molecule_energy = henergy.coul_energy + Vmol + henergy1.mol_energy
    gradient = physics.GradientNode("Gradient", (molecule_energy, positions), sign=+1)

    molecule_energy.db_name="Etot"
    gradient.db_name = "Grad"
    charge.db_name = "Mulliken"
    dipole.db_name = "dipole"
    quadrupole.db_name = "quadrupole1"
    polarizability.db_name = "polarizability"

    hierarchicality = henergy1.hierarchicality

    # define loss quantities
    from hippynn.graphs import loss

    rmse_energy = loss.MSELoss.of_node(molecule_energy) ** (1 / 2)
    rmse_grad = loss.MSELoss.of_node(gradient) ** (1 / 2)
    rmse_charge = loss.MSELoss.of_node(charge) ** (1 / 2)
    rmse_dipole = loss.MSELoss.of_node(dipole) ** (1 / 2)
    rmse_quadrupole = loss.MSELoss.of_node(quadrupole) ** (1 / 2)
    rmse_polarizability = loss.MSELoss.of_node(polarizability) ** (1 / 2)

    mae_energy = loss.MAELoss.of_node(molecule_energy)
    mae_grad = loss.MAELoss.of_node(gradient)
    mae_charge = loss.MAELoss.of_node(charge)
    mae_dipole = loss.MAELoss.of_node(dipole)
    mae_quadrupole = loss.MAELoss.of_node(quadrupole)
    mae_polarizability = loss.MAELoss.of_node(polarizability)

    rsq_energy = loss.Rsq.of_node(molecule_energy)
    rsq_grad = loss.Rsq.of_node(gradient)
    rsq_charge = loss.Rsq.of_node(charge)
    rsq_dipole = loss.Rsq.of_node(dipole)
    rsq_quadrupole = loss.Rsq.of_node(quadrupole)
    rsq_polarizability = loss.Rsq.of_node(polarizability)

    ### SLIGHTLY MORE ADVANCED USAGE

    pred_per_atom = physics.PerAtom("PeratomPredicted",(molecule_energy,species)).pred
    true_per_atom = physics.PerAtom("PeratomTrue",(molecule_energy.true,species.true))
    mae_per_atom = loss.MAELoss(pred_per_atom,true_per_atom)

    ### END SLIGHTLY MORE ADVANCED USAGE

    loss_error = (rmse_energy + mae_energy) + (rmse_grad + mae_grad) \
               + (rmse_charge + mae_charge) \
               + (rmse_dipole + mae_dipole) \
               + (rmse_quadrupole + mae_quadrupole) \
               + (rmse_polarizability + mae_polarizability)

    rbar = loss.Mean.of_node(hierarchicality)
    l2_reg = loss.l2reg(network)
    loss_regularization = 1e-6 * loss.Mean(l2_reg) + rbar    # L2 regularization and hierarchicality regularization

    train_loss = loss_error + loss_regularization

    # Validation losses are what we check on the data between epochs -- we can only train to
    # a single loss, but we can check other metrics too to better understand how the model is training.
    # There will also be plots of these things over time when training completes.
    validation_losses = {
        "T-RMSE"      : rmse_energy,
        "T-MAE"       : mae_energy,
        "T-RSQ"       : rsq_energy,
        "TperAtom MAE": mae_per_atom,
        "F-RMSE"      : rmse_grad,
        "F-MAE"       : mae_grad,
        "F-RSQ"       : rsq_grad,
        "Q-RMSE"      : rmse_charge,
        "Q-MAE"       : mae_charge,
        "Q-RSQ"       : rsq_charge,
        "Dipole-RMSE"      : rmse_dipole,
        "Dipole-MAE"       : mae_dipole,
        "Dipole-RSQ"       : rsq_dipole,
        "Quard-RMSE"      : rmse_quadrupole,
        "Quard-MAE"       : mae_quadrupole,
        "Quard-RSQ"       : rsq_quadrupole,
        "Polar-RMSE"      : rmse_polarizability,
        "Polar-MAE"       : mae_polarizability,
        "Polar-RSQ"       : rsq_polarizability,
        "T-Hier"      : rbar,
        "L2Reg"       : l2_reg,
        "Loss-Err"    : loss_error,
        "Loss-Reg"    : loss_regularization,
        "Loss"        : train_loss,
    }
    early_stopping_key = "Loss-Err"

    from hippynn import plotting

    plot_maker = plotting.PlotMaker(
        # Simple plots which compare the network to the database

        #plotting.Hist2D.compare(molecule_energy, saved=True),
        plotting.Hist2D(molecule_energy.true, molecule_energy.pred,
                        xlabel="True En",ylabel="Predicted En",
                        saved="En.pdf"),
        
        plotting.Hist2D(gradient.true, gradient.pred,
                        xlabel="True gradient",ylabel="Predicted gradient",
                        saved="gradient.pdf"),
        
        plotting.Hist2D(charge.true, charge.pred,
                        xlabel="True charge",ylabel="Predicted charge",
                        saved="charge.pdf"),
        
        plotting.Hist2D(dipole.true, dipole.pred,
                        xlabel="True dipole",ylabel="Predicted dipole",
                        saved="dipole.pdf"),
        
        plotting.Hist2D(quadrupole.true, quadrupole.pred,
                        xlabel="True quadrupole",ylabel="Predicted quadrupole",
                        saved="quadrupole.pdf"),

        plotting.Hist2D(polarizability.true, polarizability.pred,
                        xlabel="True polarizability",ylabel="Predicted polarizability",
                        saved="polarizability.pdf"),

        #Slightly more advanced control of plotting!
        plotting.Hist2D(true_per_atom,pred_per_atom,
                        xlabel="True Energy/Atom",ylabel="Predicted Energy/Atom",
                        saved="PerAtomEn.pdf"),

        plotting.HierarchicalityPlot(hierarchicality.pred,
                                     molecule_energy.pred - molecule_energy.true,
                                     saved="HierPlot.pdf"),
        plot_every=1,   # How often to make plots -- here, epoch 0, 10, 20...
    )

    if TAG==0:
        from hippynn.experiment.assembly import assemble_for_training

        training_modules, db_info = \
            assemble_for_training(train_loss,validation_losses,plot_maker=plot_maker)
        training_modules[0].print_structure()

# ----------------- #
# Step 3: RUN MODEL #
# ----------------- #

        database_params = {
            'name': dataset_name,                            # Prefix for arrays in folder
            'directory': dataset_path,
            'quiet': False,                           # Quiet==True: suppress info about loading database
            'seed': 8000,                       # Random seed for data splitting
            #'test_size': 0.01,                # Fraction of data used for testing
            #'valid_size':0.01,
            **db_info                 # Adds the inputs and targets names from the model as things to load
        }

        from hippynn.databases import DirectoryDatabase
        database = DirectoryDatabase(**database_params)
        #database.make_random_split("ignore",0.95)
        #del database.splits['ignore']
        database.make_trainvalidtest_split(test_size=0.1,valid_size=.1)

        #from hippynn.pretraining import set_e0_values
        #set_e0_values(henergy,database,energy_name="T_transpose",trainable_after=False)

        init_lr = 1e-3
        optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=init_lr)

        #TODO: 2. callback, 3. control on dataset size
        from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController

        scheduler =  RaiseBatchSizeOnPlateau(optimizer=optimizer,
                                            max_batch_size=256,
                                            patience=20,
                                            max_epochs=1000)

        controller = PatienceController(optimizer=optimizer,
                                        scheduler=scheduler,
                                        batch_size=256,
                                        eval_batch_size=256,
                                        max_epochs=1000,
                                        termination_patience=20,
                                        fraction_train_eval=0.1,
                                        stopping_key=early_stopping_key,
                                        )

        scheduler.set_batch_size_container(controller)

        experiment_params = hippynn.experiment.SetupParams(
            controller = controller,
            device='cuda',
        )
        print(experiment_params)

        # Parameters describing the training procedure.
        from hippynn.experiment import setup_training

        training_modules, controller, metric_tracker  = setup_training(training_modules=training_modules,
                                                        setup_params=experiment_params)
    if TAG>0:
        from hippynn.experiment.serialization import load_checkpoint_from_cwd, load_checkpoint
        from hippynn.experiment import train_model
        #load best model
        #structure = load_checkpoint_from_cwd()
        #load last model
        structure = load_checkpoint("experiment_structure.pt", "last_checkpoint.pt")
        training_modules = structure["training_modules"]
        database = structure["database"]
        #database.make_random_split("ignore",0.95)
        #del database.splits['ignore']
        database.make_trainvalidtest_split(test_size=0.1,valid_size=.1)
        controller = structure["controller"]
        metric_tracker = structure["metric_tracker"]

    from hippynn.experiment import train_model
    store_all_better=False
    store_best=True
    callbacks = [save_and_stop_after(training_modules, controller, metric_tracker, store_all_better, store_best, [2,0,0,0])]
    
    train_model(training_modules=training_modules,
                database=database,
                controller=controller,
                metric_tracker=metric_tracker,
                callbacks=callbacks,batch_callbacks=None,
                store_all_better=store_all_better,
                store_best=store_best)

