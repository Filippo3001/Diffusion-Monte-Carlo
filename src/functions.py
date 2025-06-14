import numpy as np
from scipy.differentiate import jacobian
import functools as ftools

def S3_pot(r):
    return (1000 * np.exp(-3  *  r**2) - 163.5 * np.exp(-1.05  *  r**2) - 21.5 * np.exp(-0.6  *  r**2)
             - 83 * np.exp(-0.8  *  r**2) - 11.5 * np.exp(-0.4  *  r**2))

# Define  the pair correlations g(r)
def Jastrow_factor(par, r):
    return np.exp(-par[0] * r**2) + par[1] * np.exp(- (par[2] + par[1]) * r**2)


def vector_distance(x, y):
    d2 = 0
    for j in len(x):
        d2 += (x[j] - y[j])**2

    return np.sqrt(d2)

# Consider x = (r_1, r_2, r_3, r_4), where  every r_i is the position of the i-nth nucleon 
def  He_trial_function(par, x):
    # Calculate the internuclear distances
    r_12 = vector_distance(x[0:3], x[3:6])
    r_13 = vector_distance(x[0:3], x[6:9])
    r_14 = vector_distance(x[0:3], x[9:12])
    r_23 = vector_distance(x[3:6], x[6:9])
    r_24 = vector_distance(x[3:6], x[9:12])
    r_34 = vector_distance(x[6:9], x[9:12])

    # The wavefunction is the given by multiplying all Jastrow factors.
    return (Jastrow_factor(par, r_12) * Jastrow_factor(par, r_13) * Jastrow_factor(par, r_14)
            * Jastrow_factor(par, r_23) * Jastrow_factor(par, r_24) * Jastrow_factor(par, r_34))

# Define a method that acts on function like kinetic part of the Hamiltonian.
# the target function need to be of the form f(par, x).
def Laplacian_op(target, param, x):
    res = 0
    jac = jacobian(ftools.partial(target, par = param), x)
    for d in jac:
        res += d**2

    return res