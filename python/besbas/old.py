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
    T = np.zeros((nbes, nbes))
    q = ikebe(l, nbes) / rcut
    dr = 0.01
    r = dr * np.arange(int(rcut/dr)+1)

    for mu in range(nbes):
        chi_mu = sphbesj(l, q[mu], r, 0)
        norm_mu = simpson(r**2 * chi_mu**2, dx=dr)
        chi_mu *= 1.0 / np.sqrt(norm_mu)

        for nu in range(nbes):
            chi_nu = sphbesj(l, q[nu], r, 0)
            dchi_nu = sphbesj(l, q[nu], r, 1)
            d2chi_nu = sphbesj(l, q[nu], r, 2)

            norm_nu = simpson(r**2 * chi_nu**2, dx=dr)
            chi_nu *= 1.0 / np.sqrt(norm_nu)
            dchi_nu *= 1.0 / np.sqrt(norm_nu)
            d2chi_nu *= 1.0 / np.sqrt(norm_nu)

            T[mu, nu] = l * (l+1) * simpson(chi_mu * chi_nu, dx=dr) \
                    - 2.0 * simpson(r * chi_mu * dchi_nu, dx=dr) \
                    - simpson(r**2 * chi_mu * d2chi_nu, dx=dr)

    return T

ecut = 100
rcut = 6.0
nbes = int(np.sqrt(ecut) * rcut / np.pi)
T = kin(0, rcut, nbes)
print(T)





