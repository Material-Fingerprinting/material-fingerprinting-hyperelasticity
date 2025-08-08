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

from material_fingerprinting.Material import Material

class Database():
    """
    An object of this class is a database of material fingerprints for multiple different materials.
    The attributes of this material are for example its parameters and their dimensions.
    The methods describe how the material responds in different experiments.
    """
    
    def __init__(self):
        
        self.db_names = []
        self.db_parameters = None
        self.db_fingerprints = None
        self.n_parameters_max = 0

    def append(self,fb):

        if fb.material.n_parameters > self.n_parameters_max:
            self.n_parameters_max = fb.material.n_parameters

        if len(self.db_names) == 0:
            self.db_parameters = np.zeros((0,self.n_parameters_max))
            self.db_fingerprints = np.zeros((0,fb.fingerprints_normalized.shape[1]))
        
        if fb.fingerprints_normalized.shape[1] != self.db_fingerprints.shape[1]:
            raise ValueError("Invalid number of data points in fingerprint.")

        self.db_names += [fb.material.name] * fb.n_fingerprints

        if fb.parameters_normalized.shape[1] == self.db_parameters.shape[1]:
            self.db_parameters = np.concatenate((self.db_parameters,fb.parameters_normalized),axis=0)
        else:
            db_parameters_pad = self.pad_array(self.db_parameters)
            parameters_normalized_pad = self.pad_array(fb.parameters_normalized)
            self.db_parameters = np.concatenate((db_parameters_pad,parameters_normalized_pad),axis=0)

        self.db_fingerprints = np.concatenate((self.db_fingerprints,fb.fingerprints_normalized),axis=0)

    def pad_array(self,array):
        if array.shape[1] < self.n_parameters_max:
            pad = np.full((array.shape[0], self.n_parameters_max - array.shape[1]), np.nan)
            return np.hstack([array, pad])
        else:
            return array

    def identify(self,measurement,_print=True):
        measurement_norm = np.linalg.norm(measurement)
        measurement_normalized = measurement / measurement_norm
        correlations = self.db_fingerprints @ measurement_normalized
        id = np.argmax(correlations)
        material = Material(self.db_names[id])
        parameters = self.db_parameters[id][~np.isnan(self.db_parameters[id])]
        parameters = material.scale_parameters(parameters,measurement_norm)
        if _print:
            print("Identifier: " + str(id))
            print("Model identified: " + self.db_names[id])
            print("Parameters identified: " + str(parameters))
            print("Formula: " + material.get_formula(parameters))
        return id, self.db_names[id], parameters

    def plot_fingerprints(self):
        for i in range(self.db_fingerprints.shape[0]):
            plt.plot(np.arange(self.db_fingerprints.shape[1]) + 1, self.db_fingerprints[i,:])
        plt.xlabel("Fingerprint Dimensions")
        plt.ylabel("Fingerprint Amplitudes")
        plt.show()

    





	
        
    
        
        
        