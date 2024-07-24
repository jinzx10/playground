from fileio import read_nao
from jlzeros import ikebe

import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson
from scipy.optimize import newton
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def quadopt(A, y, z):
    '''
    Solves for the optimal coefficients c in the quadratic optimization problem

        min (c^T A c - 2 y^T c)  subject to  c^T * diag(z) * c = 1

    '''

    # Abar_{pq} = A_{pq} / \sqrt{z_p z_q}
    Abar = A / np.sqrt(np.outer(z, z))

    # ybar = y / \sqrt{z}
    ybar = y / np.sqrt(z)

    # eigendeomposition of Abar
    e, Q = np.linalg.eigh(Abar)

    # ytilde = Q^T ybar
    ytilde = Q.T @ ybar

    # solves for the lagrange multiplier (l)
    f = lambda l: np.sum( (ytilde / (e - l))**2 ) - 1
    l = newton(f, 0)

    ctilde = ytilde / (e - l)
    cbar = Q @ ctilde
    c = cbar / np.sqrt(z)

    return Q @ (ytilde / (e - l)) / np.sqrt(z)

def projgen2(l, r, orb, rcut_proj, n_bes):

    nr_proj = np.argmax(r >= rcut_proj) + 1

    # zeros of spherical Bessel functions
    theta = ikebe(l, n_bes)

    # A matrix
    A = np.zeros((n_bes, n_bes))
    for p in range(n_bes):
        for q in range(n_bes):
            A[p, q] = theta[p] * theta[q] / rcut_proj**2  * simpson(r[:nr_proj]**2 * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj, True) * spherical_jn(l, theta[q]*r[:nr_proj]/rcut_proj, True), r[:nr_proj], even='simpson')

    # z vector
    z = 0.5 * rcut_proj**3 * spherical_jn(l+1, theta)**2

    # first order derivative of orb
    orb_spline = CubicSpline(r, orb)
    dorb = orb_spline(r, 1)

    # y vector
    y = np.zeros(n_bes)
    for p in range(n_bes):
        y[p] = simpson(r[:nr_proj]**2 * orb[:nr_proj] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj), r[:nr_proj], even='simpson') + theta[p]/rcut_proj * simpson(r[:nr_proj]**2 * dorb[:nr_proj] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj, True), r[:nr_proj], even='simpson')

    c = quadopt(A, y, z)

    # projector
    proj = np.zeros(nr_proj)
    for p in range(n_bes):
        proj += c[p] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj)

    return r[:nr_proj], proj


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
r_proj2, proj2 = projgen2(l, r, orb, rcut_proj, nbes)
#for i in range(20):
#    print('%18.12e'%proj[i])
#
#exit()

plt.plot(r, orb, label='original orbital')
plt.plot(r_proj, proj, label='projector')
plt.plot(r_proj2, proj2, label='projector2')
plt.ylim([0, 1.1 * max(np.max(orb), np.max(proj), np.max(proj2))])

plt.legend(fontsize=16)

plt.show()
