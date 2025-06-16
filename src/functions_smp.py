import numpy as np
import sympy as smp
import matplotlib as mpl
import sympy.vector as vct

def S3_pot(r):
    return (1000 * smp.exp(-3  *  r**2) - 163.5 * smp.exp(-1.05  *  r**2) - 21.5 * smp.exp(-0.6  *  r**2)
            - 83 * smp.exp(-0.8  *  r**2) - 11.5 * smp.exp(-0.4  *  r**2))

# Define  the pair correlations g(r)
def Jastrow_factor(a, beta, gamma, r):
    return smp.exp(-gamma * r**2) + a * smp.exp(- (beta + gamma) * r**2)


def vector_distance(x, y):
    d2 = 0
    for j in len(x):
        d2 += (x[j] - y[j])**2

    return smp.sqrt(d2)

# Consider x = (r_1, r_2, r_3, r_4), where  every r_i is the position of the i-nth nucleon 
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


a, beta, gamma, r = smp.symbols('a, beta, gamma, r')


print(Jastrow_factor(a, beta, gamma, r))

