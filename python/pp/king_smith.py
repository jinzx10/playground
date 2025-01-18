import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

import matplotlib.pyplot as plt

def sbt(l, frp, r, q, k=0):
    '''
    Given f(r) * r^k, compute the l-th order spherical Bessel transform
    of f(r) at q:

           / infty
    g(q) = |       dr r^(2-k) j_l(q*r) [f(r)*r^k]
           / 0

    '''
    return simpson(r**(2-k) * f * spherical_jn(l, q*r), r)


r = np.linspace(0, 10, 100)
f = r**2 * np.exp(-r*r)

q = np.linspace(0, 5, 100)
l = 2

#g_ref = np.sqrt(2)/16 * q*q * np.exp(-q*q/4)
#g = np.array([sbt(l, f, r, qi) for qi in q]) * np.sqrt(2/np.pi)

#plt.plot(q, g)
#plt.plot(q, g_ref)
#plt.show()

ecutwfc = 50
Gmax = np.sqrt(2*ecutwfc)
Gamma1 = 4 * Gmax
gamma = Gamma1 - Gmax


