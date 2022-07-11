# ---------------- #
# Imported Modules #
# ---------------- #

import os
import sys
import numpy as np
import torch
from scc_interface.scc_nodes import EEMNode, SQEPNode, PolarNode, MolTensorNode, QuadruNode, DEMNode, SplitMatrixNode
#from hippynn.interfaces.pyseqm_interface.callback import save_and_stop_after
#np.set_printoptions(threshold=np.inf)

#torch.cuda.set_device(0) # Don't try this if you want CPU training!

import matplotlib
matplotlib.use('agg')

import hippynn 

hippynn.custom_kernels.set_custom_kernels (True)

dataset_name = 'water_dimer_MP2-'    # Prefix for arrays in folder
dataset_path = '/Users/guoqingz/lanl/SCC/hippynn/scc_test/MP2'

netname = sys.argv[1]
dirname = netname
if not os.path.exists(dirname):
    os.mkdir(dirname)
else:
    pass
    #raise ValueError("Directory {} already exists!".format(dirname))
os.chdir(dirname)

TAG = int(sys.argv[2])  # False (0): first run, True(n): continue

dtype=torch.float64
torch.set_default_dtype(dtype)
if torch.cuda.is_available():
    device = torch.device('cuda')
    DEVICE = 'cuda'
else:
    device = torch.device('cpu')
    DEVICE = 'cpu'

try:
    DEVICE = [x for x in range(int(sys.argv[3]))]
except:
    pass

# Log the output of python to `training_log.txt`
with hippynn.tools.log_terminal("training_log_tag_%d.txt" % TAG,'wt'): # and torch.autograd.set_detect_anomaly(True):

    # Hyperparameters for the network

    network_params = {
        "possible_species": [0,1,8],   # Z values of the elements
        'n_features': 40,                     # Number of neurons at each layer
        "n_sensitivities": 30,                # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.6,  # qm7 1.7  qm9 .85  AL100 .85
        "dist_soft_max": 5.,  # qm7 10.  qm9 5.   AL100 5.
        "dist_hard_max": 7.5,  # qm7 15.  qm9 7.5  AL100 7.5
        "n_interaction_layers": 2,            # Number of interaction blocks
        "n_atom_layers": 3,                   # Number of atom layers in an interaction block
    }

    # Define a model

    from hippynn.graphs import inputs, networks, targets, physics

    from hippynn.experiment.serialization import load_checkpoint

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs = network_params)

    #henergy = EEMNode("EEM", network)
    #henergy = SQEPNode("SQEP", network)
    henergy = DEMNode("DEM", network)

    network1 = networks.Hipnn("HIPNN1", network.parents, module_kwargs = network_params)
    #network2 = networks.Hipnn("HIPNN", network.parents, module_kwargs = network_params)
    #henergy = SQEPNode("SEQP", (network, network2))

    henergy1 = targets.HEnergyNode("HEnergy",network1)

    ef = henergy.external_field
    dipole = henergy.dipole
    quadru = henergy.quadrupole
    #polar1 = PolarNode('Polar', (dipole, ef))

    #quadrumol = MolTensorNode("QuadruMol", network)
    #quadru = quadru1 + quadrumol
    
    #polarmol = MolTensorNode("PolarMol", network)
    polar = MolTensorNode("PolarMol", network)
    #polar = polar1 + polarmol

    molecule_energy = henergy.coul_energy + henergy1.mol_energy
    gradient = physics.GradientNode("Gradient", (molecule_energy, positions), sign=+1)

    molecule_energy.db_name = "Etot"
    dipole.db_name = "dipole2"
    #quadru.db_name = "quadrupole2"
    #polar.db_name = "polarizability1"
    gradient.db_name = "Grad"

    quadru_pred_split = SplitMatrixNode("quadru_pred_split", quadru, traceless=True)
    QpD = quadru_pred_split.diag
    QpO = quadru_pred_split.offdiag

    QpD.db_name = "quadrupole2_diag_traceless"
    QpO.db_name = "quadrupole2_offdiag"

    polar_pred_split = SplitMatrixNode("polar_pred_split", polar)
    ApD = polar_pred_split.diag
    ApO = polar_pred_split.offdiag
    ApD.db_name = "polarizability1_diag"
    ApO.db_name = "polarizability1_offdiag"

    hierarchicality = henergy1.hierarchicality

    # define loss quantities
    from hippynn.graphs import loss

    rmse_energy = loss.MSELoss.of_node(molecule_energy) ** (1 / 2)
    rmse_dipole = loss.MSELoss.of_node(dipole) ** (1 / 2)
    #rmse_quadru = loss.MSELoss.of_node(quadru) ** (1 / 2)
    #rmse_polar = loss.MSELoss.of_node(polar) ** (1 / 2)
    rmse_grad = loss.MSELoss.of_node(gradient) ** (1 / 2)
    rmse_QD = loss.MSELoss.of_node(QpD) **(1 / 2)
    rmse_QO = loss.MSELoss.of_node(QpO) ** (1 / 2)
    rmse_AD = loss.MSELoss.of_node(ApD) **(1 / 2)
    rmse_AO = loss.MSELoss.of_node(ApO) **(1 / 2)

    mae_energy = loss.MAELoss.of_node(molecule_energy)
    mae_dipole = loss.MAELoss.of_node(dipole)
    #mae_quadru = loss.MAELoss.of_node(quadru)
    #mae_polar = loss.MAELoss.of_node(polar)
    mae_grad = loss.MAELoss.of_node(gradient)
    mae_QD = loss.MAELoss.of_node(QpD)
    mae_QO = loss.MAELoss.of_node(QpO)
    mae_AD = loss.MAELoss.of_node(ApD)
    mae_AO = loss.MAELoss.of_node(ApO)

    rsq_energy = loss.Rsq.of_node(molecule_energy)
    rsq_dipole = loss.Rsq.of_node(dipole)
    #rsq_quadru = loss.Rsq.of_node(quadru)
    #rsq_polar = loss.Rsq.of_node(polar)
    rsq_grad = loss.Rsq.of_node(gradient)
    rsq_QD = loss.Rsq.of_node(QpD)
    rsq_QO = loss.Rsq.of_node(QpO)
    rsq_AD = loss.Rsq.of_node(ApD)
    rsq_AO = loss.Rsq.of_node(ApO)

    ### SLIGHTLY MORE ADVANCED USAGE

    pred_per_atom = physics.PerAtom("PeratomPredicted",(molecule_energy,species)).pred
    true_per_atom = physics.PerAtom("PeratomTrue",(molecule_energy.true,species.true))
    mae_per_atom = loss.MAELoss(pred_per_atom,true_per_atom)

    ### END SLIGHTLY MORE ADVANCED USAGE
    loss_error = (rmse_dipole + mae_dipole) + (rmse_QD + mae_QD + rmse_QO + mae_QO) + 500.0*(rmse_AD + mae_AD + rmse_AO + mae_AO) + (rmse_energy + 5.0*rmse_grad)
    #loss_error = rmse_energy + 50.0*rmse_grad

    rbar = loss.Mean.of_node(hierarchicality)
    l2_reg = loss.l2reg(network) + loss.l2reg(network1)
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
        "Dipole-RMSE"      : rmse_dipole,
        "Dipole-MAE"       : mae_dipole,
        "Dipole-RSQ"       : rsq_dipole,
        "QuadruDiag-RMSE"      : rmse_QD,
        "QuadruDiag-MAE"       : mae_QD,
        "QuadruDiag-RSQ"       : rsq_QD,
        "PolarDiag-RMSE"      : rmse_AD,
        "PolarDiag-MAE"       : mae_AD,
        "PolarDiag-RSQ"       : rsq_AD,
        "QuadruOffDiag-RMSE"      : rmse_QO,
        "QuadruOffDiag-MAE"       : mae_QO,
        "QuadruOffDiag-RSQ"       : rsq_QO,
        "PolarOffDiag-RMSE"      : rmse_AO,
        "PolarOffDiag-MAE"       : mae_AO,
        "PolarOffDiag-RSQ"       : rsq_AO,
        "F-RMSE"      : rmse_grad,
        "F-MAE"       : mae_grad,
        "F-RSQ"       : rsq_grad,
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
        
        plotting.Hist2D(dipole.true, dipole.pred,
                        xlabel="True dipole",ylabel="Predicted dipole",
                        saved="dipole.pdf"),

	plotting.Hist2D(QpD.true, QpD.pred,
                        xlabel="True quadrupole diag",ylabel="Predicted quadrupole diag",
                        saved="quadrupole_diag.pdf"),

        plotting.Hist2D(QpO.true, QpO.pred,
                        xlabel="True quadrupole offdiag",ylabel="Predicted quadrupole offdiag",
                        saved="quadrupole_offdiag.pdf"),

        plotting.Hist2D(ApD.true, ApD.pred,
                        xlabel="True polarizability diag",ylabel="Predicted polarizability diag",
                        saved="polarizability_diag.pdf"),

        plotting.Hist2D(ApO.true, ApO.pred,
                        xlabel="True polarizability offdiag",ylabel="Predicted polarizability offdiag",
                        saved="polarizability_offdiag.pdf"),

        plotting.Hist2D(gradient.true, gradient.pred,
                        xlabel="True gradient",ylabel="Predicted gradient",
                        saved="gradient.pdf"),

        #Slightly more advanced control of plotting!
        plotting.Hist2D(true_per_atom,pred_per_atom,
                        xlabel="True Energy/Atom",ylabel="Predicted Energy/Atom",
                        saved="PerAtomEn.pdf"),

        plotting.HierarchicalityPlot(hierarchicality.pred,
                                     molecule_energy.pred - molecule_energy.true,
                                     saved="HierPlot.pdf"),
        plot_every=20,   # How often to make plots -- here, epoch 0, 10, 20...
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
                                            max_batch_size=2048,
                                            patience=5)

        controller = PatienceController(optimizer=optimizer,
                                        scheduler=scheduler,
                                        batch_size=2048,
                                        eval_batch_size=2048,
                                        max_epochs=4000,
                                        termination_patience=40,
                                        fraction_train_eval=0.1,
                                        stopping_key=early_stopping_key,
                                        )

        scheduler.set_controller(controller)

        experiment_params = hippynn.experiment.SetupParams(
            controller = controller,
            device=DEVICE,
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
        structure = load_checkpoint("experiment_structure.pt", "best_checkpoint.pt")
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
    #callbacks = [save_and_stop_after(training_modules, controller, metric_tracker, store_all_better, store_best, [2,0,0,0])]
    callbacks = []
    train_model(training_modules=training_modules,
                database=database,
                controller=controller,
                metric_tracker=metric_tracker,
                callbacks=callbacks,batch_callbacks=None,
                store_all_better=store_all_better,
                store_best=store_best)

