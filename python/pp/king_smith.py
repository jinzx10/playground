import numpy as np
from scipy.special import spherical_jn, roots_legendre
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson

from sbt import sbt
from lommel import king_smith_A


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
    # VASP has an extra step that aligns qa & qb to the k-space grid.
    # Do we need this?
    
    #-------------------------------------------------------------
    #   k-space integration using Gauss-Legendre quadrature
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
            A21[i,j] = king_smith_A(l, q2[i], q1[j], R)

    A22 = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1):
            A22[i,j] = king_smith_A(l, q2[i], q2[j], R)
            A22[j,i] = A22[i,j]

    #-------------------------------------------------------------
    #           build and solve the linear system
    #-------------------------------------------------------------
    f_q1 = fq_spline(q1)
    b = np.array([np.sum(wq1 * f_q1 * A21[i]) for i in range(m)])
    A = np.pi/2 * np.diag(q2**2) - wq2 * A22
    f_q2 = np.linalg.solve(A, b)

    qtot = np.concatenate((q1, q2))
    fqtot = np.concatenate((f_q1, f_q2))
    plt.plot(q, fq, label='old')
    plt.plot(qtot, fqtot, label='new')
    plt.axhline(0.0, linestyle=':', color='k')
    plt.legend()
    plt.show()
    exit(1)

    #-------------------------------------------------------------
    #           compute the new beta(r)
    #-------------------------------------------------------------
    dr = R / nq
    r = dr * np.arange(nq)
    beta_r = np.array([
        np.sum(f_q1 * q1**2 * spherical_jn(l, q1*r[ir]) * wq1) + 
        np.sum(f_q2 * q2**2 * spherical_jn(l, q2*r[ir]) * wq2)
        for ir in range(nq)])

    return r, beta_r * (2/np.pi)


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
r_ff, beta_r_ff = king_smith_ff(l, dq, nq_cut, beta_q, qa, qb, R)

##***************************
plt.axhline(0, linestyle=':', color='k')
#plt.xlim([0, r[icut]])
plt.plot(r[:icut], rbeta[:icut]);
plt.plot(r_ff, r_ff*beta_r_ff)
plt.show()
#exit(1)
##***************************

exit(1)
#
#ecutwfc = 50; # Hartree a.u.
#Gmax = np.sqrt(2*ecutwfc);
#Gamma1 = 4 * Gmax;
#gamma = Gamma1 - Gmax;
#
#nq = 100;
#q = np.linspace(0, gamma, nq);
#dq = q[1] - q[0]
#
#iq_delim = np.argmax(q >= Gmax)
#
#q_small = q[:iq_delim]
#q_large = q[iq_delim:]
##print(q_small)
##print(q_large)
#
#
## PP_BETA in UPF is r*beta(r), not bare beta(r)
#beta_q = np.array([sbt(l, rbeta, r, qi, 1) for qi in q])
#
#####################
##plt.plot(q[:iq_delim], beta_q[:iq_delim])
##plt.plot(q, beta_q)
##plt.axhline(0, color='k', linestyle=':')
##plt.show()
##exit(1)
#####################
#
#rc = r[icut]
#R0 = 1.5 * rc;
#
#'''
#                         / R0
#    A[q,q'] = (q*q')^2 * |    dr jl(qr) * jl(q'r) * r^2
#                         / 0
#
#'''
#
#from lommel import king_smith_A
#
#A = np.zeros((nq, nq));
#for i in range(nq):
#    for j in range(i+1):
#        A[i,j] = king_smith_A(l, q[i], q[j], R0)
#        A[j,i] = A[i,j]
#
##plt.imshow(A)
#
#b0 = dq * A[iq_delim:, :iq_delim] @ beta_q[:iq_delim]
#
#b = np.zeros(len(q_large))
#for iql, ql in enumerate(q_large):
#    b[iql] = simpson(A[iq_delim + iql, :iq_delim] * beta_q[:iq_delim], x = q_small)
#
##print('b = ', b)
##print('b0 = ', b0)
##print(np.linalg.norm(b-b0))
##plt.plot(q_large, b)
##plt.plot(q_large, b0)
##plt.show()
##
##exit(1)
#B = np.pi/2 * np.diag(q_large**2) - dq * A[iq_delim:, iq_delim:]
#y = np.linalg.solve(B, b0)
#
#beta_q_new = np.copy(beta_q)
#beta_q_new[iq_delim:] = y
#
##print(q[iq_delim-1])
##print(q[iq_delim])
##print(f'Gmax = {Gmax}, gamma = {gamma}')
##exit(1)
#
plt.plot(q, beta_q, label='old')
plt.plot(q, beta_q_new, label = 'new')
plt.axhline(0.0, linestyle=':')
plt.axvline(Gmax, linestyle='--')
plt.legend()

plt.show()



