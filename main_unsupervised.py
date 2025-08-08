"""
                           
 _|      _|      _|_|_|_|  
 _|_|  _|_|      _|        
 _|  _|  _|      _|_|_|    
 _|      _|      _|        
 _|      _|  _|  _|    _|  
                           
 Material        Fingerprinting

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

database_path = 'fenicsx/data/database_unsupervised11_30200_normalized.npz'
exp_paths = [
    'fenicsx/data/experiment_unsupervised11_Blatz_noise0.0.npz',
    # 'fenicsx/data/experiment_unsupervised11_Blatz_noise0.001.npz',
    'fenicsx/data/experiment_unsupervised11_Blatz_noise0.01.npz',
    'fenicsx/data/experiment_unsupervised11_Blatz_noise0.05.npz',

    'fenicsx/data/experiment_unsupervised11_Demiray_noise0.0.npz',
    # 'fenicsx/data/experiment_unsupervised11_Demiray_noise0.001.npz',
    'fenicsx/data/experiment_unsupervised11_Demiray_noise0.01.npz',
    'fenicsx/data/experiment_unsupervised11_Demiray_noise0.05.npz',

    'fenicsx/data/experiment_unsupervised11_MooneyRivlin_noise0.0.npz',
    # 'fenicsx/data/experiment_unsupervised11_MooneyRivlin_noise0.001.npz',
    'fenicsx/data/experiment_unsupervised11_MooneyRivlin_noise0.01.npz',
    'fenicsx/data/experiment_unsupervised11_MooneyRivlin_noise0.05.npz',

    'fenicsx/data/experiment_unsupervised11_NeoHooke_noise0.0.npz',
    # 'fenicsx/data/experiment_unsupervised11_NeoHooke_noise0.001.npz',
    'fenicsx/data/experiment_unsupervised11_NeoHooke_noise0.01.npz',
    'fenicsx/data/experiment_unsupervised11_NeoHooke_noise0.05.npz',

]

def formula(p):
    W = "W = "
    if not np.isclose(p[1],0.0):
        W += f"{p[1]:.2f} [\\bar I_1 - 3] + "
    if not np.isclose(p[2],0.0):
        W += f"{p[2]:.2f} [\\bar I_1 - 3]^2 + "
    if not np.isclose(p[3],0.0):
        W += f"{p[3]:.2f} [\\bar I_1 - 3]^3 + "
    if not np.isclose(p[4],0.0):
        W += f"{p[4]:.2f} [\\bar I_2 - 3] + "
    if not np.isclose(p[5],0.0):
        W += f"{p[5]:.2f} [\\exp({p[8]:.2f} [\\bar I_1 - 3]) - 1] + "
    if not np.isclose(p[6],0.0):
        W += f"{p[6]:.2f} [\\exp({p[9]:.2f} [\\bar I_2 - 3]^2) - 1] + "
    if not np.isclose(p[7],0.0):
        W += f" - {p[7]:.2f} [\\ln(1 - {p[10]:.2f} [\\bar I_1 - 3])] + "
    W += f"{p[0]:.2f} [J - 1]^2"
    return W

def strain_energy_density_lam(p,lam1,lam2,lam3):
    I1 = lam1**2 + lam2**2 + lam3**2
    I2 = lam1**2 * lam2**2 + lam2**2 * lam3**2 + lam1**2 * lam3**2
    J = lam1 * lam2 * lam3
    I1_bar = J**(-2.0/3.0) * I1
    I2_bar = J**(-4.0/3.0) * I2
    W = p[0] * (J - 1)**(2.0)
    W += p[1] * (I1_bar - 3)
    W += p[2] * (I1_bar - 3)**(2.0)
    W += p[3] * (I1_bar - 3)**(3.0)
    W += p[4] * (I2_bar - 3)
    W += p[5] * (np.exp(p[8] * (I1_bar - 3)) - 1.0) # Demiray
    W += p[6] * (np.exp(p[9] * (I1_bar - 3)**(2.0)) - 1.0) # Holzapfel
    W += - p[7] * (np.log(1.0 - p[10] * (I1_bar - 3))) # Gent
    return W

def error_strain_energy_density_compressible_lam(parameters_true,parameters_disc,epsabs=1e-4,epsrel=1e-4):
    x_lower = 0.75
    x_upper = 1.25
    y_lower = lambda x: x_lower
    y_upper = lambda x: x_upper
    z_lower = lambda x, y: x_lower
    z_upper = lambda x, y: x_upper

    def integrand_W_diff(lam1,lam2,lam3):
        W_true = strain_energy_density_lam(parameters_true,lam1,lam2,lam3)
        W_disc = strain_energy_density_lam(parameters_disc,lam1,lam2,lam3)
        return abs(W_true - W_disc)
    
    def integrand_W_true(lam1,lam2,lam3):
        W_true = strain_energy_density_lam(parameters_true,lam1,lam2,lam3)
        return abs(W_true)
    
    integral_diff, _ = scipy.integrate.tplquad(integrand_W_diff, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, epsabs=epsabs, epsrel=epsrel)
    integral_W_true, _ = scipy.integrate.tplquad(integrand_W_true, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper, epsabs=epsabs, epsrel=epsrel)

    return integral_diff / integral_W_true

def error_strain_energy_density_incompressible_lam(parameters_true,parameters_disc,epsabs=1e-4,epsrel=1e-4):
    x_lower = 0.75
    x_upper = 1.25
    y_lower = lambda x: x_lower
    y_upper = lambda x: x_upper

    def integrand_W_diff(lam1,lam2):
        lam3 = 1.0 / (lam1 * lam2)
        W_true = strain_energy_density_lam(parameters_true,lam1,lam2,lam3)
        W_disc = strain_energy_density_lam(parameters_disc,lam1,lam2,lam3)
        return abs(W_true - W_disc)
    
    def integrand_W_true(lam1,lam2):
        lam3 = 1.0 / (lam1 * lam2)
        W_true = strain_energy_density_lam(parameters_true,lam1,lam2,lam3)
        return abs(W_true)
    
    integral_diff, _ = scipy.integrate.dblquad(integrand_W_diff, x_lower, x_upper, y_lower, y_upper, epsabs=epsabs, epsrel=epsrel)
    integral_W_true, _ = scipy.integrate.dblquad(integrand_W_true, x_lower, x_upper, y_lower, y_upper, epsabs=epsabs, epsrel=epsrel)

    return integral_diff / integral_W_true

def identify(database_path,exp_path,measure = "INNER",lam = 1.0,VERBOSE = True):

    """
    load database
    """
    database = np.load(database_path)
    database_parameters = database['parameters']
    database_f_reaction = database['f_reaction']
    database_f_u = database['f_u']
    IDX_HOMOGENEITY = database['IDX_HOMOGENEITY']
    IDX_NONHOMOGENEITY = database['IDX_NONHOMOGENEITY']

    """
    load experimenatal data
    """
    exp = np.load(exp_path)
    exp_parameters_true = exp['parameters']
    exp_f_reaction = exp['f_reaction']
    exp_f_u = exp['f_u']

    """
    pattern recognition
    """
    f_reaction_norm = np.linalg.norm(exp_f_reaction)
    exp_f_reaction_normalized = exp_f_reaction / f_reaction_norm

    if measure == "INNER":
        database_f_u_normalized = database_f_u / np.linalg.norm(database_f_u, axis=1, keepdims=True)
        exp_f_u_normalized = exp_f_u / np.linalg.norm(exp_f_u)
        measure = database_f_reaction @ exp_f_reaction_normalized + lam * database_f_u_normalized @ exp_f_u_normalized
    elif measure == "INNER & NORM":
        measure = database_f_reaction @ exp_f_reaction_normalized - lam * np.linalg.norm(database_f_u - exp_f_u, axis=1)
    elif measure == "NORM":
        measure = np.linalg.norm(database_f_reaction - exp_f_reaction_normalized, axis=1) + lam * np.linalg.norm(database_f_u - exp_f_u, axis=1)

    measure[np.isnan(measure)] = -np.inf
    id = np.argmax(measure)
    parameters_discovered = np.concatenate([database_parameters[id,IDX_HOMOGENEITY] * f_reaction_norm, database_parameters[id,IDX_NONHOMOGENEITY]])

    if VERBOSE:
        print(" ")
        print("Experiment: " + exp_path)
        print("True parameters: " + str(exp_parameters_true))
        print("Discovered parameters: " + str(parameters_discovered))
        print("True energy: " + formula(exp_parameters_true))
        print("Discovered energy: " + formula(parameters_discovered))
        error_compressible = error_strain_energy_density_compressible_lam(exp_parameters_true, parameters_discovered)
        print("Error (compressible): " + "{:.2e}".format(error_compressible))
        error_incompressible = error_strain_energy_density_incompressible_lam(exp_parameters_true, parameters_discovered)
        print("Error (incompressible): " + "{:.2e}".format(error_incompressible))
        # print("Discovered identifier: " + str(id))
        # print("Discovered measure: " + str(measure[id]))

for exp_path in exp_paths:
    identify(database_path,exp_path)