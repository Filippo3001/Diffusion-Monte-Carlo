import numpy as np
import sympy as smp
import matplotlib as mpl
import vector as vct

def S3_pot(r):
    return (1000 * smp.exp(-3  *  r**2) - 163.5 * smp.exp(-1.05  *  r**2) - 21.5 * smp.exp(-0.6  *  r**2)
            - 83 * smp.exp(-0.8  *  r**2) - 11.5 * smp.exp(-0.4  *  r**2))

#Define the symbols representing the parameters(a, beta, gamma) and the 4 position vectors.
a, beta, gamma = smp.symbols('a, beta, gamma')
r1x, r1y, r1z, r2x, r2y, r2z, r3x, r3y, r3z, r4x, r4y, r4z = smp.symbols('r1:5(x:z)')

# Construct the vectors
r1 = vct.VectorSympy3D(x = r1x, y = r1y, z = r1z)
r2 = vct.VectorSympy3D(x = r2x, y = r2y, z = r2z)
r3 = vct.VectorSympy3D(x = r3x, y = r3y, z = r3z)
r4 = vct.VectorSympy3D(x = r4x, y = r4y, z = r4z)

# Calculate the internuclear distances
r_12 = r1 - r2
r_13 = r1 - r3
r_14 = r1 - r4
r_23 = r2 - r3
r_24 = r2 - r4
r_34 = r3 - r4

# Construct the potential, which is the sum of the S3 pot for all internuclear distances
Potential = (S3_pot(abs(r_12)) + S3_pot(abs(r_13)) + S3_pot(abs(r_14)) + S3_pot(abs(r_23)) + S3_pot(abs(r_24)) + S3_pot(abs(r_34)))

# Define  the pair correlations g(r)
def Jastrow_factor(a, beta, gamma, r):
    return smp.exp(-gamma * r**2) + a * smp.exp(- (beta + gamma) * r**2)

# Construct the He_trial_function 
He_trial_function = (Jastrow_factor(a, beta, gamma, abs(r_12)) * Jastrow_factor(a, beta, gamma, abs(r_13)) 
                    * Jastrow_factor(a, beta, gamma, abs(r_14)) * Jastrow_factor(a, beta, gamma, abs(r_23)) 
                    * Jastrow_factor(a, beta, gamma, abs(r_24)) * Jastrow_factor(a, beta, gamma, abs(r_34)))

# The laplacian operator is applied to a function in reference to a 3D vector
def Laplacian(f, r):
    return smp.diff(f, r.x, 2) + smp.diff(f, r.y, 2) + smp.diff(f, r.z, 2)

# Evaluate the Hamiltonian over the 4 position vectors
kin_part = Laplacian(He_trial_function, r1) + Laplacian(He_trial_function, r2) + Laplacian(He_trial_function, r3) + Laplacian(He_trial_function, r4)
Hamiltonian = kin_part + Potential * He_trial_function

# We finally arrive at the local energy
local_en = Hamiltonian / He_trial_function

# They seem too slow to evaluate to use. Need to use numerical methods.
