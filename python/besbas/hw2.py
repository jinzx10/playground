from jlzeros import ikebe
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

nbes0 = 35
l = 2
zeros = ikebe(l, nbes0)
rcut = 7
q = np.array([zero / rcut for zero in ikebe(l, nbes0)])

ecut = 100
nbes = np.argmax(q*q > ecut)
#nbes = nbes0

q = q[:nbes]
zeros = zeros[:nbes]
print(nbes)

def fac(n, iq):
    return np.sqrt(2) / (rcut**1.5 * np.abs(spherical_jn(n+1, zeros[iq])))

def jn(n, iq, r, deriv):
    if deriv == 0:
        return fac(n, iq) * spherical_jn(n, q[iq]*r)
    else:
        if n == 0:
            return -fac(n, iq) * q[iq] * jn(1, iq, r, deriv-1) / fac(1, iq)
        else:
            return  ( n * jn(n-1, iq, r, deriv-1)/fac(n-1, iq) - (n+1) * jn(n+1, iq, r, deriv-1)/fac(n+1, iq) ) \
                    * fac(n, iq) * q[iq] / (2 * n + 1)

dr = 0.01
r = dr * np.arange(0, int(rcut/dr) + 1)

raw = np.zeros((len(r), nbes))
for iq in range(nbes):
    raw[:, iq] = jn(l, iq, r, 0)

plot_raw = False
if plot_raw:
    fig, ax = plt.subplots()
    for iq in range(nbes):
        ax.plot(r, raw[:,iq])
    
    plt.show()
    exit()

T_raw = np.diag(q**2)
print(q**2)
exit()

M = 4

# the 1, 2, ..., M th derivatives of the spherical Bessel function at rcut
dmat = np.zeros((M, nbes))
for iq in range(nbes):
    for deriv in range(M):
        dmat[deriv, iq] = jn(l, iq, rcut, deriv+1)

u, s, vh = np.linalg.svd(dmat, full_matrices=True)
print(s)
C = vh.T[:,M:]
#print(np.linalg.norm(dmat @ C, np.inf))
print(dmat[0]/dmat[1])
print(dmat[2]/dmat[3])
#u, s, vh = np.linalg.svd(dmat[:1,:], full_matrices=True)
#C = vh.T[:,1:]
#print(np.linalg.norm(dmat @ C, np.inf))
exit()

new = raw @ C
plot_new = False
if plot_new:
    fig, ax = plt.subplots()
    for i in range(new.shape[1]):
    #for i in range(0, 5):
        ax.plot(r, new[:,i])
    
    plt.show()
    exit()


# kinetic matrix of new
nbes_new = nbes - M
T_new = C.T @ T_raw @ C

val, vec = np.linalg.eigh(T_new)
print(val)

new2 = new @ vec
plot_new2 = True
if plot_new2:
    fig, ax = plt.subplots()
    for i in range(0, min(5, new2.shape[1])):
        ax.plot(r, new2[:,i])
    
    plt.show()




