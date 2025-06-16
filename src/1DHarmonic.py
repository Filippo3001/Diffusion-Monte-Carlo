import os
import sys

sys.path.append(os.path.abspath("/home/filippo/Github/Variational-Monte-Carlo/src/classes"))  # Change the path to a more universal one

import numpy as np
import matplotlib.pyplot as plt
import functools as fct
from functions import Laplacian_op

# We want to compare analytical results with result of Montecarlo implementation
def trial_state(alpha, x):
    return (np.sqrt(alpha)  /  np.pow(np.pi, 1/4)) * np.exp(- 0.5 * alpha**2 * x**2)

alpha2 = 0.5

def trial_state2(x):
    return (np.sqrt(alpha2)  /  np.pow(np.pi, 1/4)) * np.exp(- 0.5 * alpha2**2 * x**2)

def Hamiltonian(f, x):
    return - Laplacian_op(f, x) + x**2 * f(x)

def local_en(alpha, x):
    return Hamiltonian(fct.partial(trial_state, alpha = alpha), x) / trial_state(alpha, x)

def local_en2(x):
    return Hamiltonian(trial_state2, x) / trial_state(alpha, x)

def importance_sampling(alpha, x):
    return trial_state(alpha, x)**2

# We now define the analytical result

def local_en_an(alpha, x):
    return alpha**2 + x**2 * (1 - alpha**4)

def results_an(alpha):
    mean = 0.5 * (alpha**2 + 1 /  alpha**2)
    var = (alpha**4 - 1)**2 / (2 * alpha**4)

alpha = 0.5

x = np.linspace([-5], [5], 100)
print(x)
y = local_en2(x)
print('test')

fig, ax = plt.subplots()

ax.plot(x, y)

plt.show()