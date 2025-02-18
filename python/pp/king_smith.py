import numpy as np
from scipy.special import spherical_jn, roots_legendre
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson

from lommel import lommel_j
from jlzeros import JLZEROS
from sbt import sbt

def king_smith_ff(l, dq, nq, fq, qa, qb, R):
    r'''
    Fourier filtering via the method proposed by King-Smith et al.

    Given a k-space radial function f(q) tabulated on a uniform grid

                    0, dq, 2*dq, ..., (nq-1)*dq

    this subroutine looks for a new k-space piecewise radial function

               / f(q)       q <= qa
        g(q) = | h(q)       qa < q < qb
               \  0         q >= qb

    where h(q) is determined by minimizing the "tail" of G(r) outside
    the cutoff radius R:

            /+inf
        I = |     dr (r*G(r))^2
            / R

    here G(r) is the spherical Bessel transform of g(q).

    Parameters
    ----------

    l : int
        angular momentum quantum number
    dq : float
        k-space radial grid spacing
    nq : int
        number of k-space radial grid points
    fq : array
        values on the k-space radial grid
    qa, qb : float
        endpoints of the k-space interval where optimization is performed
    R : float
        r-space cutoff radius

    '''
    assert(qa < qb)
    assert(qa <= (nq-1)*dq)

    q = dq * np.arange(nq)
    fq_spline = CubicSpline(q, fq, extrapolate=False)

    # TODO
    # VASP has an extra step that aligns qa & qb to the k-space grid,
    # but I do not see the point of that. Do we need this?
    
    #-------------------------------------------------------------
    #               Gauss-Legendre quadrature
    #-------------------------------------------------------------
    m = 50 # quadrature order     NOTE: VASP use 32
    roots, weights = roots_legendre(m)

    # to transform an integration from [-1, -1] to [a, b]:
    # x = roots * (b-a)/2 + (a+b)/2
    # w = weights * (b-a)/2

    # [0, qa]
    q1 = roots * qa / 2 + qa / 2
    wq1 = weights * qa / 2

    # [qa, qb]
    q2 = roots * (qb - qa) / 2 + (qb + qa) / 2
    wq2 = weights * (qb - qa) / 2

    #-------------------------------------------------------------
    #               tabulate A_l(q,q',R)
    #-------------------------------------------------------------
    # A21: q \in [qa, qb], q' \in [0, qa]
    # A22: q, q' \in [qa, qb]

    A21 = np.zeros((m, m));
    for i in range(m):
        for j in range(m):
            A21[i,j] = (q2[i]*q1[j])**2 * lommel_j(l, q2[i], q1[j], R)

    A22 = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1):
            A22[i,j] = (q2[i]*q2[j])**2 * lommel_j(l, q2[i], q2[j], R)
            A22[j,i] = A22[i,j]

    #-------------------------------------------------------------
    #           build and solve the linear system
    #-------------------------------------------------------------
    f_q1 = fq_spline(q1)
    b = np.array([np.sum(wq1 * f_q1 * A21[i]) for i in range(m)])
    A = np.pi/2 * np.diag(q2**2) - wq2 * A22
    f_q2 = np.linalg.solve(A, b)

    #qtot = np.concatenate((q1, q2))
    #fqtot = np.concatenate((f_q1, f_q2))
    #plt.plot(q, fq, label='old')
    #plt.plot(qtot, fqtot, label='new')
    #plt.axhline(0.0, linestyle=':', color='k')
    #plt.legend()
    #plt.show()
    #exit(1)

    #-------------------------------------------------------------
    #       compute the new beta(r) on the linear grid
    #-------------------------------------------------------------
    dr = R / nq
    r = dr * np.arange(nq)
    fr = np.array([
        np.sum(f_q1 * q1**2 * spherical_jn(l, q1*r[ir]) * wq1) + 
        np.sum(f_q2 * q2**2 * spherical_jn(l, q2*r[ir]) * wq2)
        for ir in range(nq)])
    fr *= 2.0 / np.pi

    #-------------------------------------------------------------
    #               inspect r-space "tail"
    #-------------------------------------------------------------
    # SBT should be self-inverse, but a function cannot be strictly
    # localized in both r and k space. Since the new beta is made
    # truncated in k-space, it must have some tail in the r-space.
    # We can estimate the tail by inspecting the error from two
    # consecutive SBTs: f(q) -> f(r) -> f2(q).
    # The difference between f(q) and f2(q) results from the
    # truncation in r-space.

    # SBT f(q) -> f(r) towards an r-space quadrature grid
    r_quad = roots * R / 2 + R / 2
    wr_quad = weights * R / 2
    fr_quad = np.array([
        np.sum(f_q1 * q1**2 * spherical_jn(l, q1*r_quad[ir]) * wq1) + 
        np.sum(f_q2 * q2**2 * spherical_jn(l, q2*r_quad[ir]) * wq2)
        for ir in range(m)])
    fr_quad *= 2.0/np.pi

    # SBT f(r) -> f2(q) towards the previous k-space grid
    qtot = np.concatenate((q1, q2))
    f2q = np.array([
        np.sum(wr_quad * fr_quad * r_quad**2
               * spherical_jn(l, qtot[iq]*r_quad))
        for iq in range(2*m)
    ])

    fq = np.concatenate((f_q1, f_q2))
    err = np.linalg.norm(fq - f2q, np.inf)

    #plt.plot(qtot, fq)
    #plt.plot(qtot, f2q)
    #plt.show()
    #exit(1)

    return r, fr, err


def opt_sphbes(l, dq, nq, fq, qa, qb, R, nbes, alpha):
    r'''
    Optimized superposition of truncated spherical Bessel functions.

    Let theta[m] be the m-th positive zero of the l-th order spherical
    Bessel function, z[m] the m-th spherical Bessel function with node
    at R:

                      /  jl(theta(m)*r/R)     r <= R
            z[m](r) = |
                      \        0              r >  R

    Given a k-space radial function f(q) tabulated on a uniform grid

                    0, dq, 2*dq, ..., (nq-1)*dq

    this subroutine looks for a new function g(r) as a superposition
    of z[m]:

                    g(r) = sum(c[k] * z[k](r))
                            k

    such that

            / qa                          / +inf
    alpha * |    dq q^2 (f(q) - h(q))^2 + |      dq q^2 h(q)^2
            / 0                           / qb

    is minimized. Here h(q) is the spherical Bessel transform of g(r)
    and alpha is a relative weight factor.

    '''
    assert(qa < qb)
    assert(qa <= (nq-1)*dq)

    q = dq * np.arange(nq)
    fq_spline = CubicSpline(q, fq, extrapolate=False)

    #-------------------------------------------------------------
    #   wave vectors that satisfies the boundary condition
    #-------------------------------------------------------------
    q_bc = JLZEROS[l][:nbes] / R

    #-------------------------------------------------------------
    #               Gauss-Legendre quadrature
    #-------------------------------------------------------------
    m = 64 # quadrature order
    roots, weights = roots_legendre(m)

    # to transform an integration from [-1, -1] to [a, b]:
    # x = roots * (b-a)/2 + (a+b)/2
    # w = weights * (b-a)/2

    # [0, qa]
    q1 = roots * qa / 2 + qa / 2
    wq1 = weights * qa / 2

    #-------------------------------------------------------------
    #           spherical Bessel transform of z[k](r)
    #-------------------------------------------------------------
    # Z[k][i] is the k-th transformed function at q1[i]
    Z = np.zeros((nbes, m))
    for k in range(nbes):
        for i in range(m):
            Z[k,i] = lommel_j(l, q_bc[k], q1[i], R)




import xml.etree.ElementTree as ET

import time
import matplotlib.pyplot as plt

# get some actual PP_BETA from pseudopotential file
#pp_file = '/home/zuxin/downloads/pseudopotentials/SG15/PBE/Si_ONCV_PBE-1.0.upf'
pp_file = '/home/zuxin/downloads/pseudopotentials/SG15/PBE/Au_ONCV_PBE-1.0.upf'
tree = ET.parse(pp_file)
root = tree.getroot()

for pp_r in root.iter('PP_R'):
    r = np.array([float(x) for x in pp_r.text.split()])


for pp_beta1 in root.iter('PP_BETA.5'):
    l = int(pp_beta1.attrib['angular_momentum'])
    rbeta = np.array([float(x) for x in pp_beta1.text.split()])

    # icut in UPF file may not be reliable (?)
    # e.g., the index is 270 for PP_BETA.1 in Ag_ONCV_PBE-1.0.upf
    # which truncates the data at a non-zero value.
    #icut = int(pp_beta1.attrib['cutoff_radius_index'])

# find the index of the first trailing zero
icut = len(r) - 1
while rbeta[icut] == 0:
    icut -= 1
icut += 1

R = r[icut] * 1.5

# k-space radial grid
qcut = 20
dq = 0.01
nq_cut = int(qcut / dq) + 1;
q = dq * np.arange(nq_cut)

#print(rbeta[icut-1])
#print(rbeta[icut])
#

beta_q = np.array([sbt(l, rbeta, r, qi, k=1) for qi in q])

#beta_q_max = np.array([sbt(l, rbeta, r, qi) for qi in qtot])
#plt.plot(qtot, beta_q_max)
#plt.xlim([0, Gmax])
#plt.axhline(0.0, linestyle=':')
#plt.show()
#exit(1)

qa = 8
qb = 15
r_ff, beta_r_ff, err_ff = king_smith_ff(l, dq, nq_cut, beta_q, qa, qb, R)

##***************************
plt.axhline(0, linestyle=':', color='k')
#plt.xlim([0, r[icut]])
plt.plot(r[:icut], rbeta[:icut]);
plt.plot(r_ff, r_ff*beta_r_ff)
plt.show()
#exit(1)
##***************************

exit(1)
