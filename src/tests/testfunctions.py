import numpy as np
import matplotlib.pyplot as plt
from scipy.differentiate import hessian

def testfunctions(x):
    return x[0] + x[1]**2 - 4 * x[2]*3

def testfunctions2(x):
    return x**3

print(hessian(testfunctions2, [1,2]))
