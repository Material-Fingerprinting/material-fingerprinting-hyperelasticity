# -*- coding: utf-8 -*-
"""
Created on Thu May 11 18:50:13 2023

@author: flaschel
"""
# import third party packages
import _pickle as pickle
import scipy.io

def save_object(objectname,filename):
    filename = filename + ".pickle"        
    pickle_out = open(filename, "wb") 
    pickle.dump(objectname,pickle_out)
    pickle_out.close()
    
def load_object(filename):
    filename = filename + ".pickle"
    pickle_in = open(filename, "rb")
    objectname = pickle.load(pickle_in)
    pickle_in.close()
    return objectname