import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

import matplotlib.pyplot as plt

from jlzeros import ikebe

def sphbesj(l, q, r, deriv):
    if deriv == 0:
        return spherical_jn(l, q*r)
    elif deriv == 1:
        return q * spherical_jn(l, q*r, True)
    else:
        if l == 0:
            return - q * sphbesj(1, q, r, deriv-1)
        else:
            return q * ( l * sphbesj(l-1, q, r, deriv-1) - (l+1) * sphbesj(l+1, q, r, deriv-1) ) / (2 * l + 1)

def kin(l, rcut, nbes):
    return ( ikebe(l, nbes) / rcut )**2

ecut = 100
rcut = 6.0
nbes = int(np.sqrt(ecut) * rcut / np.pi)
T = kin(0, rcut, nbes)
print(T)





