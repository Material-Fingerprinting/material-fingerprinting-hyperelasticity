"""
                           
 _|      _|      _|_|_|_|  
 _|_|  _|_|      _|        
 _|  _|  _|      _|_|_|    
 _|      _|      _|        
 _|      _|  _|  _|    _|  
                           
 Material        Fingerprinting

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import material_fingerprinting as mf
import subprocess

np.random.seed(0)

## === Load database ====
exp1 = mf.Experiment(mode="uniaxial tension - finite strain")
exp2 = mf.Experiment(mode="simple shear - finite strain")
exp_union = mf.ExperimentUnion([exp1,exp2])

fp1 = mf.Fingerprints(mf.Material(name="Blatz-Ko - incompressible"),exp_union)
fp2 = mf.Fingerprints(mf.Material(name="Demiray - incompressible"),exp_union)
fp3 = mf.Fingerprints(mf.Material(name="Gent - incompressible"),exp_union)
fp4 = mf.Fingerprints(mf.Material(name="Holzapfel - incompressible"),exp_union)
fp5 = mf.Fingerprints(mf.Material(name="Mooney-Rivlin - incompressible"),exp_union)
fp6 = mf.Fingerprints(mf.Material(name="Neo-Hooke - incompressible"),exp_union)
fp7 = mf.Fingerprints(mf.Material(name="Ogden - incompressible"),exp_union)

db = mf.Database()
db.append(fp1)
db.append(fp2)
db.append(fp3)
db.append(fp4)
db.append(fp5)
db.append(fp6)
db.append(fp7)

## === Test fingerprinting ====
# Define the models and their parameters
models = {
    'Blatz-Ko - incompressible': np.array([50.0]),
    'Demiray - incompressible': np.array([10.0, 8.0]),
    'Mooney-Rivlin - incompressible': np.array([10.0, 40.0]),
    'Neo-Hooke - incompressible': np.array([10.0]),
    'Ogden - incompressible': np.array([5.0, 8.0]),
    }

## === Conduct experiments and identify models with and without noise ====
noise_001 = 0.01
noise_005 = 0.05

for model, param in models.items():
    print("Model true: " + model)
    model_abbrev = ''.join(e for e in model if e.isalnum())
    model_abbrev = model_abbrev.replace('incompressible', '')

    mat = mf.Material(name=model) # true material
    measurement = mat.conduct_experiment_union(exp_union,parameters = param)
    noisy_measurement1 = measurement + np.random.normal(loc=0.0, scale=noise_001 * np.max(np.abs(measurement)), size=measurement.shape)
    noisy_measurement2 = measurement + np.random.normal(loc=0.0, scale=noise_005 * np.max(np.abs(measurement)), size=measurement.shape)

    _, model_type, param_no_noise = db.identify(measurement.T)
    mat_no_noise = mf.Material(name=model_type)
    error = mf.get_error_strain_energy_density_incompressible_lam(mat, param, mat_no_noise, param_no_noise)
    print("Error: " + "{:.2e}".format(error))
    measurement_id = mat_no_noise.conduct_experiment_union(exp_union,parameters = param_no_noise)

    _, model_type, param_noise_001 = db.identify(noisy_measurement1.T)
    mat_noise_001 = mf.Material(name=model_type)
    error = mf.get_error_strain_energy_density_incompressible_lam(mat, param, mat_noise_001, param_noise_001)
    print("Error: " + "{:.2e}".format(error))
    measurement_id_noise_001 = mat_noise_001.conduct_experiment_union(exp_union,parameters = param_noise_001)

    _, model_type, param_noise_005 = db.identify(noisy_measurement2.T)
    mat_noise_005 = mf.Material(name=model_type)
    error = mf.get_error_strain_energy_density_incompressible_lam(mat, param, mat_noise_005, param_noise_005)
    print("Error: " + "{:.2e}".format(error))
    measurement_id_noise_005 = mat_noise_005.conduct_experiment_union(exp_union,parameters = param_noise_005)

    print(" ")
