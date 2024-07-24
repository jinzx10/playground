import numpy as np
import itertools as it


# lattice vectors 
a = np.random.randn(3, 3)

# Reciprocal lattice vectors (column-wise)
b = np.linalg.inv(a).T * np.pi * 2

# check that b is indeed the reciprocal lattice vectors
# a[:,i] \dot b[:,j] = 2\pi \delta_{ij}
assert np.linalg.norm(a.T @ b - 2.*np.pi*np.eye(3), np.inf) < 1e-12

# lattice vectors (in units of a)
Ra1 = range(-1,3)
Ra2 = range(0,5)
Ra3 = range(2,8)

# lattice vectors
R_in_a = list(it.product(Ra1, Ra2, Ra3))
R_in_a = np.array([list(r) for r in R_in_a]).T
R = a @ R_in_a

nR1 = len(Ra1)
nR2 = len(Ra2)
nR3 = len(Ra3)
nR = nR1 * nR2 * nR3

# matrix size
sz = 3

# k sampling
nk1 = nR1 + 4
nk2 = nR2 + 2
nk3 = nR3 + 3
nk = nk1 * nk2 * nk3

kb1 = np.linspace(-0.5, 0.5, nk1, endpoint=False)
kb2 = np.linspace(-0.5, 0.5, nk2, endpoint=False)
kb3 = np.linspace(-0.5, 0.5, nk3, endpoint=False)

# shift k to become a symmetric grid (optional)
dkb1 = kb1[1] - kb1[0]
dkb2 = kb2[1] - kb2[0]
dkb3 = kb3[1] - kb3[0]

kb1 += 0.5 * dkb1
kb2 += 0.5 * dkb2
kb3 += 0.5 * dkb3

k_in_b = list(it.product(kb1, kb2, kb3))
k_in_b = np.array([list(k) for k in k_in_b]).T
k = b @ k_in_b


# H(R)
H_R = np.random.randn(nR, sz, sz)

# H(k)
H_k = np.zeros((nk, sz, sz), dtype=complex)
for ik in range(nk):
    for iR in range(nR):
        H_k[ik] += H_R[iR] * np.exp(-1j * np.dot(k[:,ik],R[:,iR]))

# recover H(R)
H_R2 = np.zeros((nR, sz, sz), dtype=complex)
for iR in range(nR):
    for ik in range(nk):
        H_R2[iR] += H_k[ik] * np.exp(1j * np.dot(R[:,iR], k[:,ik])) / nk

assert np.all([np.linalg.norm(H_R[iR] - H_R2[iR], np.inf) < 1e-12 for iR in range(nR)])

