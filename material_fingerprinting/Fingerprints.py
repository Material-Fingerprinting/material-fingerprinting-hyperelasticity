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

class Fingerprints():
    """
    An object of this class is a database of material fingerprints for one material.
    The attributes of this database are for example the number of fingerprints, the material and the considered experiments.
    The methods describe for example rules how the fingerprints should be generated.
    """
    
    def __init__(self,material,experiment,n_fingerprints=100):
        
        self.material = material
        self.experiment = experiment
        self.n_fingerprints = n_fingerprints
        self.set_parameters()
        self.compute_fingerprints()
        self.normalize()

    def set_parameters(self):
        range_positive = np.linspace(0.1,10,self.n_fingerprints)
        range_greater_one = np.linspace(1,10,self.n_fingerprints)
        range_smaller_one = np.linspace(0.1,1,self.n_fingerprints)
        # we may also set parameters on a logarithmic scale
        # self.parameters[:,1] = np.logspace(-1,1,self.n_fingerprints)

        if self.material.n_parameters == 1:
            self.n_fingerprints = 1
            self.parameters = np.array([[1.0]])
        elif self.material.n_parameters == 2:
            self.parameters = np.zeros((self.n_fingerprints,self.material.n_parameters))
            self.parameters[:,0] = np.ones(self.n_fingerprints)
            if self.material.name == "Demiray - incompressible":
                self.parameters[:,1] = range_positive
            elif self.material.name == "Gent - incompressible":
                self.parameters[:,1] = range_smaller_one
            elif self.material.name == "Holzapfel - incompressible":
                self.parameters[:,1] = range_positive
            elif self.material.name == "Mooney-Rivlin - incompressible":
                self.parameters[:,1] = range_positive
            elif self.material.name == "Ogden - incompressible":
                self.parameters[:,1] = range_positive
            elif self.material.name == "Yeoh quadratic - incompressible":
                self.parameters[:,1] = range_positive
            else:
                raise ValueError("Parameter ranges are not implemented for this material.")

    def compute_fingerprints(self):
        self.fingerprints = np.zeros((self.n_fingerprints,self.experiment.n_steps))
        if self.experiment.n_experiment == 1:
            for i in range(self.n_fingerprints):
                self.fingerprints[i,:] = self.material.conduct_experiment(self.experiment,parameters=self.parameters[i,:])
        elif self.experiment.n_experiment > 1:
            for i in range(self.n_fingerprints):
                self.fingerprints[i,:] = self.material.conduct_experiment_union(self.experiment,parameters=self.parameters[i,:])
            # for exp in self.experiment.experiment_list:
            #     for i in range(self.n_fingerprints):
            #         self.fingerprints[i,exp.fingerprint_idx[0]:exp.fingerprint_idx[1]] = self.material.conduct_experiment(exp,parameters=self.parameters[i,:])
        
    def normalize(self):
        # Assumption: The fingerprints have the physical dimension of a force.
        # We choose the unit of the force, such that the fingerprints are normalized.
        fingerprint_norms = np.linalg.norm(self.fingerprints, axis=1, keepdims=True)
        self.parameters_normalized = self.parameters
        for i in range(self.material.n_parameters):
            if self.material.dim_parameters[i] == "dimensionless":
                self.parameters_normalized[:,i] = self.parameters[:,i]
            elif self.material.dim_parameters[i] == "force/area":
                self.parameters_normalized[:,i] = self.parameters[:,i] / fingerprint_norms[:,0]
            else:
                raise ValueError("This parameter dimension is not implemented.")
        self.fingerprints_normalized = self.fingerprints / fingerprint_norms
    
    def plot_fingerprints(self,normalized=False):

        if normalized:
            y_data = self.fingerprints_normalized
        else:
            y_data = self.fingerprints

        for i in range(self.n_fingerprints):
            plt.plot(self.experiment.control, y_data[i])
        
        if self.experiment.n_experiment == 1:
            plt.xlabel(self.experiment.control_str[0])
            plt.ylabel(self.experiment.measurement_str[0])
        else:
            plt.xlabel("Fingerprint Dimensions")
            plt.ylabel("Fingerprint Amplitudes")

        plt.show()
        
    def delete(self):
        for attr in list(self.__dict__):
            delattr(self, attr)
        del self








	
        
    
        
        
        