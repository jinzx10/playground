from jlzeros import ikebe, bracket_d
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spherical_jn

def jn(l, q, r, deriv):
    if deriv == 0:
        return spherical_jn(l, q*r)
    elif deriv == 1:
        return q * spherical_jn(l, q*r, 1)
    else:
        if l == 0:
            return - q * jn(1, q, r, deriv-1)
        else:
            return q * ( l * jn(l-1, q, r, deriv-1) - (l+1) * jn(l+1, q, r, deriv-1) ) / (2 * l + 1)

l = 2
rcut = 7
nbes = 20

q = [zero / rcut for zero in ikebe(l, nbes)]

dr = 0.01
r = dr * np.arange(0, int(rcut/dr) + 1)

raw = np.zeros((len(r), nbes))
for iq in range(nbes):
    raw[:, iq] = spherical_jn(l, q[iq]*r)

plot_raw = False
if plot_raw:
    fig, ax = plt.subplots()
    for iq in range(nbes):
        ax.plot(r, raw[:,iq])
    
    plt.show()

M = 1

# the 1, 2, ..., M th derivatives of the spherical Bessel function at rcut
dmat = np.zeros((M, nbes))
for iq in range(nbes):
    for deriv in range(M):
        dmat[deriv, iq] = jn(l, q[iq], rcut, deriv+1)

u, s, vh = np.linalg.svd(dmat, full_matrices=True)
C = vh.T[:,M:]
print(np.linalg.norm(dmat @ C, np.inf))

new = raw @ C
plot_new = True
if plot_new:
    fig, ax = plt.subplots()
    #for i in range(new.shape[1]):
    for i in range(5, 10):
        ax.plot(r, new[:,i])
    
    plt.show()




