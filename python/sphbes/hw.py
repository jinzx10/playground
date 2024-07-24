import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson
import matplotlib.pyplot as plt

def smooth(r, rcut, sigma):
    if abs(sigma) < 1e-14:
        g = np.ones_like(r)
    else:
        g = 1. - np.exp(-0.5*((r-rcut)/sigma)**2)

    g[r >= rcut] = 0.0
    return g



rcut = 6
dr = 0.01
r = dr * np.arange(int(rcut/dr)+1)
nbes = 20

zeros = np.pi * np.arange(1, nbes+1)

def inner_prod(r, f, g):
    return simpson(f * g * r**2, r)

def coeff(p, r, rcut, sigma):
    g = smooth(r, rcut, sigma)
    f = spherical_jn(0, zeros[p]*r/rcut)
    return inner_prod(r, f, g) * 2.0 / (rcut**3 * spherical_jn(1, zeros[p])**2)


sigma = 2.0
# recover the smoothing function from a combination of spherical bessel functions
g2 = sum(coeff(i, r, rcut, sigma) * spherical_jn(0, zeros[i]*r/rcut) for i in range(nbes))

plt.plot(r, smooth(r, rcut, sigma), label='original')
plt.plot(r, g2, label='reconstructed')
plt.legend()
plt.show()




exit()
def sphbes_forward_recurrence(l, x):
    if l == 0:
        return np.sin(x) / x
    elif l == 1:
        return np.sin(x) / x**2 - np.cos(x) / x
    else:
        return (2*l-1) / x * sphbes_forward_recurrence(l-1, x) - sphbes_forward_recurrence(l-2, x)


l = 22
n = 500
err = np.zeros(n)

mid = 2. * l / np.exp(1)
w = mid - 0.001

x = np.linspace(mid-w, mid+w, n)

j_recur = sphbes_forward_recurrence(l, x)
j_exact = spherical_jn(l, x)

err = np.abs(j_recur - j_exact)

plt.plot(x, np.log10(err))
plt.axvline(x = mid, color='black', label='2l/e')
plt.axvline(x = l/2, color = 'red',  label='l/2')
plt.legend()
plt.show()





