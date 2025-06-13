import sys
import os

sys.path.append(os.path.abspath("/home/filippo/Github/Variational-Monte-Carlo/src/classes"))

import numpy as np
import matplotlib.pyplot as plt
import Montecarlo as Mc

#test the generate function to see if it can reproduce the target importance sampling

mean = 2
sigma = 0.5

def importance_sampl(x):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp( (-1/2) * ((x - mean) / sigma)**2)   #it is a gauss distribution

#Plot the distribution

x_axis = np.linspace(0, 4, 1000)
y_axis = importance_sampl(x_axis)

fig, ax = plt.subplots()

ax.plot(x_axis, y_axis)

#Now generate a histogram to confront

mc = Mc.MonteCarlov1(importance_sampl)

points = mc.generate(0, 10000, 50, 0.1)

n_bins  = 40

hist, ax2 = plt.subplots()

ax2.hist(points, n_bins)

plt.show()
