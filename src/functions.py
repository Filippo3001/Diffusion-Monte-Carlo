import numpy as np
from scipy.differentiate import hessian
import functools as ftools
import random as rd

#This is the non simpy version, using vector

def S3_pot(r):
    return (1000 * np.exp(-3  *  r**2) - 163.5 * np.exp(-1.05  *  r**2) - 21.5 * np.exp(-0.6  *  r**2)
             - 83 * np.exp(-0.8  *  r**2) - 11.5 * np.exp(-0.4  *  r**2))

def He_pot(r1, r2, r3, r4):
    r_12 = r1 - r2
    r_13 = r1 - r3
    r_14 = r1 - r4
    r_23 = r2 - r3
    r_24 = r2 - r4
    r_34 = r3 - r4

    return S3_pot(r_12) + S3_pot(r_13) + S3_pot(r_14) + S3_pot(r_23) + S3_pot(r_24) + S3_pot(r_34)

# Define the pair correlations g(r)
def Jastrow_factor(a, beta, gamma, r):
    return np.exp(-gamma * r**2) + a * np.exp(- (beta + gamma) * r**2)


def vector_module(x):
    mod2 = 0
    for i in x:
        mod2 += i**2
    return np.sqrt(mod2)


# Use for ri 1D numpy arrays with of lenght 3(or 2D arrays if the function need to be evaluated over multiple points) They represent position vectors
def  He_trial_function(a, beta, gamma, r1, r2, r3, r4):
    # Calculate the internuclear distances
    r_12 = r1 - r2
    r_13 = r1 - r3
    r_14 = r1 - r4
    r_23 = r2 - r3
    r_24 = r2 - r4
    r_34 = r3 - r4

    # The wavefunction is the given by multiplying all Jastrow factors.
    return ( Jastrow_factor(a, beta, gamma, vector_module(r_12)) * Jastrow_factor(a, beta, gamma, vector_module(r_13)) 
           * Jastrow_factor(a, beta, gamma, vector_module(r_14)) * Jastrow_factor(a, beta, gamma, vector_module(r_23)) 
           * Jastrow_factor(a, beta, gamma, vector_module(r_24)) * Jastrow_factor(a, beta, gamma, vector_module(r_34)))

# Define a method that acts on function like kinetic part of the Hamiltonian.
# the target function need to be of the form f(x), where x is a numpy array
def Laplacian_op(target, x):
    hess = hessian(target, x)
    res = np.trace(hess.ddf)
    return res


def He_Hamiltonian(target, r1, r2, r3, r4):
    kin_part = ( Laplacian_op(ftools.partial(target, r2 = r2, r3 = r3, r4 = r4), r1) + Laplacian_op(ftools.partial(target, r1 = r1, r3 = r3, r4 = r4), r2)
               + Laplacian_op(ftools.partial(target, r1 = r1, r2 = r2, r4 = r4), r3) + Laplacian_op(ftools.partial(target, r1 = r1, r2 = r2, r3 = r3), r4))
    
    pot_part = He_pot(r1, r2, r3, r4) * target(r1, r2, r3, r4)
    return - kin_part + pot_part