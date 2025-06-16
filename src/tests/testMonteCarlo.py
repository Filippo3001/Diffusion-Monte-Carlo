import sys
import os

sys.path.append(os.path.abspath("/home/filippo/Github/Varational-Monte-Carlo/src/classes"))

import numpy as np
import unittest
from Montecarlo import MonteCarlov1
# Use as trial function the function with alpha of 1D Harmonic oscillator

alpha = 0.5

def trial_state(x):
    return (np.sqrt(alpha)/(np.pi**(1/4))) * np.exp((- alpha**2 *  x**2)/2)

def trial_state2(x):
    return trial_state(x)**2

def local_en(x):
    return alpha**2 + (x**2) * (1 - alpha**4)
    

mc = MonteCarlov1(trial_state2)

result = mc.integrate(local_en, 0, 10000, 10, 0.1)

print(result)