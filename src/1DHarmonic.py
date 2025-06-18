import os
import sys

sys.path.append(os.path.abspath("/home/filippo/Github/Varational-Monte-Carlo/src/classes"))  # Change the path to a more universal one

import numpy as np
import matplotlib.pyplot as plt
import functools as fct
from functions import Laplacian_op
from Montecarlo import MonteCarlov2

# We want to compare analytical results with result of Montecarlo implementation
def trial_state(x, alpha):
    return (np.sqrt(abs(alpha))  /  np.pow(np.pi, 1/4)) * np.exp(- 0.5 * alpha**2 * x**2)

def Hamiltonian(f, x):
    return - Laplacian_op(f, x) + x**2 * f(x)

def local_en(alpha, x):
    return Hamiltonian(fct.partial(trial_state, alpha = alpha), x) / trial_state(x, alpha)

def importance_sampling(x, alpha):
    return trial_state(x, alpha)**2

# We now define the analytical result

def local_en_an(alpha, x):
    return alpha**2 + x**2 * (1 - alpha**4)

def results_an(alpha):
    mean = 0.5 * (alpha**2 + 1 /  alpha**2)
    var = (alpha**4 - 1)**2 / (2 * alpha**4)

alpha = 0.5

#print(local_en2(np.array([0.])))

#x = np.linspace([-5.], [5.], 100)

x = np.array([np.linspace(-5, 5, 100)])

#print(x)
y = importance_sampling(alpha, x)
#y_an = local_en_an(alpha, x)

mc = MonteCarlov2(fct.partial(importance_sampling, alpha = alpha), np.array([2.]))

mc.generate(10000, 100, 0.04)

#print(np.concatenate(mc.accepted_points))ot(*x, *y)
#ax.plot(*x, *y_an)

#a

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

ax.plot(*x, *y)
#ax.plot(*x, *y_an)

ax2.hist(np.concatenate(mc.accepted_points), 200)

plt.show()