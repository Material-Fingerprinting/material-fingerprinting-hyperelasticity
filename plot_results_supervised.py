"""
                           
 _|      _|      _|_|_|_|  
 _|_|  _|_|      _|        
 _|  _|  _|      _|_|_|    
 _|      _|      _|        
 _|      _|  _|  _|    _|  
                           
 Material        Fingerprinting

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import material_fingerprinting as mf
import pickle
import subprocess

np.random.seed(0)

plt.style.use('plots/paper_style.mplstyle')
try:
    with open('plots/colors.pickle', 'rb') as handle:
        colors = pickle.load(handle)
except:
    subprocess.run(["python", "plots/colors.py"])
    with open('plots/colors.pickle', 'rb') as handle:
        colors = pickle.load(handle)
        
show = True
save = True
close = True
textwidth = 6.5 # inches

## STYLE PARAMETERS ================
data_col = 'black'
model_col = colors['Mooney-Rivlin']
model_linestyle = '--'
model_linewidth = 1.5
scatter_size = 15
# ===============================

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

    ## === Plot fingerprints for all three noise levels ====
    fig, ax = plt.subplots(1, 3, sharex=True)
    fig.suptitle(f"{model}", fontsize=10)
    xvec = np.arange(measurement.shape[1])

    ax[0].plot(xvec, measurement_id.squeeze(), color=model_col, linestyle=model_linestyle, linewidth=model_linewidth, label='Discovered')
    ax[0].scatter(xvec, measurement.squeeze(), color=data_col, label='Data', s=scatter_size)

    ax[1].plot(xvec, measurement_id_noise_001.squeeze(), color=model_col, linestyle=model_linestyle, linewidth=model_linewidth, label='Discovered')
    ax[1].scatter(xvec, noisy_measurement1.squeeze(), color=data_col, label='Data', s=scatter_size)

    ax[2].plot(xvec, measurement_id_noise_005.squeeze(), color=model_col, linestyle=model_linestyle, linewidth=model_linewidth, label='Discovered')
    ax[2].scatter(xvec, noisy_measurement2.squeeze(), color=data_col, label='Data', s=scatter_size)

    for a in ax:
        # a.legend()
        a.grid(True)
        a.minorticks_on() 
        a.grid(True, which='minor', linestyle='--', color='lightgray', linewidth=0.5)
        # a.set_ylabel('$\\mathbf{f}^{(i)}$')
        # a.set_xlabel("$i$")
        fig.set_size_inches(textwidth, 0.35*textwidth)
    fig.tight_layout()
    if show: plt.show(block=False)
    if save: fig.savefig(f'plots/fingerprints_{model_abbrev}.png', bbox_inches='tight')
    if close: plt.close(fig)


    ## === Plot stress strain for NOISE 2 ====
    # !!! brute force color change !!!
    color_stress_strain = colors['Ogden'] if model_abbrev == 'NeoHooke' else colors[model_abbrev]

    fig1, ax1 = plt.subplots(1,2)
    # plot results for uniaxial tension
    indices = np.arange(exp1.n_steps)
    ax1[0].scatter(exp1.control,noisy_measurement2.squeeze()[indices], color=data_col, s=scatter_size, label='Data')
    ax1[0].plot(exp1.control, mat_noise_005.conduct_experiment(exp1, parameters=param_noise_005).squeeze(), color=color_stress_strain, linestyle=model_linestyle, linewidth=model_linewidth, label='Discovered')
    ax1[0].set_xlabel(exp1.control_str[0])
    ax1[0].set_ylabel(exp1.measurement_str[0])
    ax1[0].set_title("Uniaxial Tension", fontsize=10)
    ax1[0].legend()
    # plot results for simple shear
    indices = np.arange(exp2.n_steps) + exp1.n_steps
    ax1[1].scatter(exp2.control,noisy_measurement2.squeeze()[indices], color=data_col, s=scatter_size, label='Data')
    ax1[1].plot(exp2.control,mat_noise_005.conduct_experiment(exp2,parameters = param_noise_005).squeeze(), color=color_stress_strain, linestyle=model_linestyle, linewidth=model_linewidth, label='Discovered')
    ax1[1].set_xlabel(exp2.control_str[0])
    ax1[1].set_ylabel(exp2.measurement_str[0])
    ax1[1].set_title("Simple Shear", fontsize=10)
    for a in ax1:
        a.grid(True)
        a.minorticks_on() 
        a.grid(True, which='minor', linestyle='--', color='lightgray', linewidth=0.5)
    fig1.set_size_inches(textwidth, 0.35*textwidth)
    fig1.tight_layout()
    if show: plt.show(block=False)
    if save: fig1.savefig(f'plots/stress_strain_noise2_{model_abbrev}.png', bbox_inches='tight', dpi=1000)
    if close: plt.close(fig1)

    

