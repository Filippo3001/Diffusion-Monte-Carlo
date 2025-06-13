import numpy as np
import matplotlib.pyplot as plt

def S3_pot(r):
    return (1000 * np.exp(-3  *  r**2) - 163.5 * np.exp(-1.05  *  r**2) - 21.5 * np.exp(-0.6  *  r**2)
             - 83 * np.exp(-0.8  *  r**2) - 11.5 * np.exp(-0.4  *  r**2))

x_axis =  np.linspace(0.5, 5, 1000)
y_axis =  S3_pot(x_axis)

fig, ax = plt.subplots()

ax.plot(x_axis, y_axis)

plt.show()