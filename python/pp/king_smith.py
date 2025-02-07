import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

from sbt import sbt
from lommel import king_smith_A

def king_smith_ff(r, rbeta_r, Gcut, Gmax, R):

    # k-space radial grid
    nq = 100;
    q = np.linspace(0, Gmax, nq);
    dq = q[1] - q[0]
    
    iq_delim = np.argmax(q >= Gmax)
    
    q_small = q[:iq_delim]
    q_large = q[iq_delim:]

    beta_q = np.array([sbt(l, rbeta, r, qi, 1) for qi in q])

    A = np.zeros((nq, nq));
    for i in range(nq):
        for j in range(i+1):
            A[i,j] = king_smith_A(l, q[i], q[j], R0)
            A[j,i] = A[i,j]

    b0 = dq * A[iq_delim:, :iq_delim] @ beta_q[:iq_delim]
    
    b = np.zeros(len(q_large))
    for iql, ql in enumerate(q_large):
        b[iql] = simpson(A[iq_delim + iql, :iq_delim] * beta_q[:iq_delim], x = q_small)

    B = np.pi/2 * np.diag(q_large**2) - dq * A[iq_delim:, iq_delim:]
    y = np.linalg.solve(B, b0)
    
    beta_q_new = np.copy(beta_q)
    beta_q_new[iq_delim:] = y

    return q, beta_q, beta_q_new


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

#print(rbeta[icut-1])
#print(rbeta[icut])

##***************************
#plt.plot(r[:icut], rbeta[:icut]);
#plt.axhline(0, linestyle=':', color='k')
#plt.xlim([0, r[icut]])
#plt.show()
#exit(1)
##***************************

Gcut = 20
Gmax = 40
R = 5
q, beta_q, beta_q_new = king_smith_ff(r[:icut], rbeta[:icut], Gcut, Gmax, R)

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



