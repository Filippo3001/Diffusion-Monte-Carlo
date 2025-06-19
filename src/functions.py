import numpy as np
from scipy.differentiate import hessian
import functools as ftools
import random as rd


# Define a method that acts on function like the Laplacian operator.
# The target function need to be of the form f(x), where x is a numpy array
def Laplacian_op(target, x):
    hess = hessian(target, x)
    res = np.trace(hess.ddf)
    return res

# We define all functions relative to the 1D Harmonic oscillator

def harm_trial_state(x, alpha):
    return (np.sqrt(abs(alpha))  /  np.pow(np.pi, 1/4)) * np.exp(- 0.5 * alpha**2 * x**2)

def harm_Hamiltonian(f, x):
    return - Laplacian_op(f, x) + x**2 * f(x)

def local_en(x, alpha):
    return harm_Hamiltonian(ftools.partial(harm_trial_state, alpha = alpha), x) / harm_trial_state(x, alpha)

def harm_importance_sampling(x, alpha):
    return harm_trial_state(x, alpha)**2

# We now define the analytical result for the Harmonic oscillator

def harm_local_en_an(x, alpha):
    return alpha**2 + x**2 * (1 - alpha**4)

def harm_results_an(alpha):
    mean = 0.5 * (alpha**2 + 1 /  alpha**2)
    var = (alpha**4 - 1)**2 / (2 * alpha**4)

# We define all functions relative to the Helium nucleus

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
           

def He_Hamiltonian(target, r1, r2, r3, r4):
    kin_part = ( Laplacian_op(ftools.partial(target, r2 = r2, r3 = r3, r4 = r4), r1) + Laplacian_op(ftools.partial(target, r1 = r1, r3 = r3, r4 = r4), r2)
               + Laplacian_op(ftools.partial(target, r1 = r1, r2 = r2, r4 = r4), r3) + Laplacian_op(ftools.partial(target, r1 = r1, r2 = r2, r3 = r3), r4))
    
    pot_part = He_pot(r1, r2, r3, r4) * target(r1, r2, r3, r4)
    return - kin_part + pot_part