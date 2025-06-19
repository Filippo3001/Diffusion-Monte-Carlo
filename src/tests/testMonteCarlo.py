import sys
import os

sys.path.append(os.path.abspath("/home/filippo/Github/Varational-Monte-Carlo/src/classes"))

sys.path.append(os.path.abspath("/home/filippo/Github/Varational-Monte-Carlo/src"))

import numpy as np
import matplotlib.pyplot as plt
import functools as fct
from functions import *
from Montecarlo import MonteCarlov2

alpha = 0.5

# Use as trial function the function with alpha of 1D Harmonic oscillator

#print(local_en2(np.array([0.])))

#x = np.linspace([-5.], [5.], 100)

x = np.array([np.linspace(-5, 5, 100)])

#print(x)
y = harm_importance_sampling(x, alpha)
#y_an = local_en_an(alpha, x)

mc = MonteCarlov2(fct.partial(harm_importance_sampling, alpha = alpha), np.array([2.]))

mc.generate(10000, 100, 0.2)

#print(np.concatenate(mc.accepted_points))ot(*x, *y)
#ax.plot(*x, *y_an)

mean, sigma = mc.evaluate(fct.partial(local_en, alpha = alpha))

for m, s in zip(mean, sigma):
    print(m, ' +/- ', s)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

ax.plot(*x, *y)
#ax.plot(*x, *y_an)

ax2.hist(np.concatenate(mc.accepted_points), 100)

plt.show()