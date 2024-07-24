import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

mu = 0.0                     # Order mu of Bessel function
r = np.logspace(-7, 1, 128)  # Input evaluation points
dln = np.log(r[1]/r[0])      # Step size
offset = fft.fhtoffset(dln, initial=-6*np.log(10), mu=mu)
k = np.exp(offset)/r[::-1]   # Output evaluation points

def f(x, mu):
    """Analytical function: x^(mu+1) exp(-x^2/2)."""
    return x**(mu + 1)*np.exp(-x**2/2)

a_r = f(r, mu)
fht = fft.fht(a_r, dln, mu=mu, offset=offset)

a_k = f(k, mu)
rel_err = abs((fht-a_k)/a_k)

rgrid = np.exp(r)

plt.plot(rgrid, a_r)
plt.show()
