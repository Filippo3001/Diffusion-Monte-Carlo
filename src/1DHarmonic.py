import os
import sys

sys.path.append(os.path.abspath("/home/filippo/Github/Variational-Monte-Carlo/src/classes"))  # Change the path to a more universal one

import numpy as np
import matplotlib as mpl
from functions import Laplacian_op

# We want to compare analytical results with result of Montecarlo implementation
def trial_state(par, x):
    return (np.sqrt(par)  /  np.pow(np.pi, 1/4)) * np.exp(- 0.5 * par**2 * x**2)

def Hamiltonian(f, par, x):
    return - Laplacian_op(f, par, x) + x**2 * f(par, x)

def local_en(par, x):
    return Hamiltonian(trial_state, par, x) / trial_state(par, x)

# We now define the analytical result

def local_en_an(par, x):
    return par**2 + x**2 * (1 - par**4)

def results_an(par):
    mean = 0.5 * (par**2 + 1 /  par**2)
    var = (par**4 - 1)**2 / (2 * par**4)