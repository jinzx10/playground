from fileio import read_nao
from jlzeros import ikebe

import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson
import matplotlib.pyplot as plt

def projgen(l, r, orb, rcut_proj, n_bes):

    nr_proj = np.argmax(r >= rcut_proj) + 1

    # zeros of spherical Bessel functions
    theta = ikebe(l, n_bes)

    # z vector
    z = 0.5 * rcut_proj**3 * spherical_jn(l+1, theta)**2

    # w vector
    w = np.zeros(n_bes)
    for p in range(n_bes):
        w[p] = simpson(r[:nr_proj]**2 * orb[:nr_proj] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj), r[:nr_proj], even='simpson')

    # optimal coefficients
    prefac = 1.0 / np.sqrt(np.sum(w*w/z))
    c = prefac * w / z

    # projector
    proj = np.zeros(nr_proj)
    for p in range(n_bes):
        proj += c[p] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj)

    return r[:nr_proj], proj


# read numerical atomic orbital
#nao = read_nao('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/26_Fe_DZP/Fe_gga_7au_100Ry_4s2p2d1f.orb')
#dr = nao['dr']
#chi = nao['chi']
#rcut_nao = nao['rcut']
#r = dr * np.arange(len(chi[0][0]))
#
## target orbital
#l = 2
#zeta = 0
#orb = chi[l][zeta]

# test orbital
l = 2
r = np.linspace(0, 10, 1001);
orb = r * r * np.exp(-r)
N = simpson((orb * r)**2, r)
orb *= 1.0 / np.sqrt(N)

# rcut for the projector
rcut_proj = 7
nbes = 7

r_proj, proj = projgen(l, r, orb, rcut_proj, nbes)
for i in range(20):
    print('%18.12e'%proj[i])

exit()

plt.plot(r, orb, label='original orbital')
plt.plot(r_proj, proj, label='projector')
plt.ylim([0, 1.1 * max(np.max(orb), np.max(proj))])

plt.legend(fontsize=16)

plt.show()
