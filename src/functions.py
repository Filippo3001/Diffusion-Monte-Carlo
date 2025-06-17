import numpy as np
from scipy.differentiate import hessian
import functools as ftools
import vector as vct
import random as rd

#This is the non simpy version, using vector

def S3_pot(r):
    return (1000 * np.exp(-3  *  r**2) - 163.5 * np.exp(-1.05  *  r**2) - 21.5 * np.exp(-0.6  *  r**2)
             - 83 * np.exp(-0.8  *  r**2) - 11.5 * np.exp(-0.4  *  r**2))

# Define  the pair correlations g(r)
def Jastrow_factor(a, beta, gamma, r):
    return np.exp(-gamma * r**2) + a * np.exp(- (beta + gamma) * r**2)


def vector_distance(x, y):
    d2 = 0
    for j in len(x):
        d2 += (x[j] - y[j])**2

    return np.sqrt(d2)


# Use for ri vector library 3d geometric vectors
def  He_trial_function(a, beta, gamma, r1, r2, r3, r4):
    # Calculate the internuclear distances
    r_12 = r1 - r2
    r_13 = r1 - r3
    r_14 = r1 - r4
    r_23 = r2 - r3
    r_24 = r2 - r4
    r_34 = r3 - r4

    # The wavefunction is the given by multiplying all Jastrow factors.
    return (Jastrow_factor(a, beta, gamma, abs(r_12)) * Jastrow_factor(a, beta, gamma, abs(r_13)) 
            * Jastrow_factor(a, beta, gamma, abs(r_14)) * Jastrow_factor(a, beta, gamma, abs(r_23)) 
            * Jastrow_factor(a, beta, gamma, abs(r_24)) * Jastrow_factor(a, beta, gamma, abs(r_34)))

# Define a method that acts on function like kinetic part of the Hamiltonian.
# the target function need to be of the form f(x), where x is a vector(or list)
def Laplacian_op(target, x):
    res = 0
    hess = hessian(target, x)
    for i in hess:
        res += hess[i,i]**2            #check if the implementation makes sense. IT DOES NOT, ptobably better to use sympy

    return res