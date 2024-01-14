from fileio import read_nao
from jlzeros import ikebe

import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson
from scipy.optimize import newton, brentq
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def diagquadsolve(D, y):
    '''
    Solves for the optimal coefficients c in the diagonal quadratic optimization problem

        min (c^T diag(D) c - 2 y^T c)  subject to  c^T * c = 1

    '''
    D = np.sort(D)
    sz = len(D)

    # bracketing intervals for root finding
    w = [D[0] - np.sqrt(np.sum(y*y)), *D, D[-1] + np.sqrt(np.sum(y*y))]

    f = lambda l: np.sum( (y / (D - l))**2 ) - 1

    f_tmp = 1e300
    l_tmp = None



def quadsolve(A, y, z):
    '''
    Solves for the optimal coefficients c in the quadratic optimization problem

        min (c^T A c - 2 y^T c)  subject to  c^T * diag(z) * c = 1

    '''

    # Abar_{pq} = A_{pq} / \sqrt{z_p z_q}
    Abar = A / np.sqrt(np.outer(z, z))

    # ybar = y / \sqrt{z}
    ybar = y / np.sqrt(z)

    # eigen-decomposition of Abar
    e, Q = np.linalg.eigh(Abar)

    # ytilde = Q^T ybar
    ytilde = Q.T @ ybar

    # solves for the lagrange multiplier (l)
    f = lambda l: np.sum( (ytilde / (e - l))**2 ) - 1
    #l = newton(f, 0)
    l = brentq(f, e[0] - np.linalg.norm(ytilde), e[0]-1e-10)
    #l = brentq(f, e[-1]+1e-10, e[-1] + np.linalg.norm(ytilde) )

    ####
    print(e)
    print(l)
    print(f(l))
    #nl = 1000
    #ll = np.linspace(-5, 20, nl)
    #fl = np.zeros(nl)
    #for il in range(nl):
    #    fl[il] = f(ll[il])
    #plt.plot(ll, fl)
    #plt.ylim([-1, 1])
    #plt.show()
    ####


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
        #y[p] = simpson(r[:nr_proj]**2 * orb[:nr_proj] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj), r[:nr_proj], even='simpson') \
        #        + simpson(r[:nr_proj]**2 * dorb[:nr_proj] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj, True), r[:nr_proj], even='simpson') * theta[p]/rcut_proj
        y[p] = simpson(r[:nr_proj]**2 * dorb[:nr_proj] * spherical_jn(l, theta[p]*r[:nr_proj]/rcut_proj, True), r[:nr_proj], even='simpson') * theta[p]/rcut_proj

    c = quadsolve(A, y, z)

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
rcut_proj = 6
nbes = 10

r_proj, proj = projgen(l, r, orb, rcut_proj, nbes)
r_proj2, proj2 = projgen2(l, r, orb, rcut_proj, nbes)
#for i in range(20):
#    print('%18.12e'%proj[i])
#
#exit()

plt.plot(r, orb, label='original orbital')
plt.plot(r_proj, proj, label='projector (overlap)')
plt.plot(r_proj2, proj2, label='projector (overlap+gradient)')
plt.ylim([0, 1.1 * max(np.max(orb), np.max(proj), np.max(proj2))])

plt.legend(fontsize=16)

plt.show()
