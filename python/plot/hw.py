import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)

f = 1. / (np.exp(x) + 1.)
fermi = -(f * np.log(f) + (1. - f) * np.log(1. - f))
gauss = np.exp(-x*x) / np.sqrt(np.pi)

plt.plot(x,fermi, label='fermi')
plt.plot(x,gauss, label='gauss')

plt.legend()
plt.show()
