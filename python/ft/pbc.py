import numpy as np
import itertools as it
import matplotlib.pyplot as plt

# Bravais lattice vectors (column-wise)
a = np.random.randn(3,3)

# Reciprocal lattice vectors (column-wise)
b = np.linalg.inv(a).T * np.pi * 2

# check that b is indeed the reciprocal lattice vectors
# ai \dot bj = 2\pi \delta_{ij}
assert np.linalg.norm(a.T @ b - 2.*np.pi*np.eye(3), np.inf) < 1e-12

# lattice vectors (in units of a)
Ra1 = range(-1,3)
Ra2 = range(0,5)
Ra3 = range(2,8)
n1 = len(Ra1)
n2 = len(Ra2)
n3 = len(Ra3)
N = n1 * n2 * n3

R_in_a = list(it.product(Ra1, Ra2, Ra3))
R_in_a = np.array([list(r) for r in R_in_a]).T

# lattice vectors
R = a @ R_in_a

#fig = plt.figure(figsize=(12, 12))
#ax = fig.add_subplot(projection='3d')
#ax.scatter(R[0,:], R[1,:], R[2,:])

# PBC-compatible k-sampling (in units of b)
Rb1 = np.arange(n1) / n1
Rb2 = np.arange(n2) / n2
Rb3 = np.arange(n3) / n3

k_in_b = list(it.product(Rb1, Rb2, Rb3))
k_in_b = np.array([list(k) for k in k_in_b]).T

k = b @ k_in_b

# check
print(np.linalg.norm(np.exp(1j * N * k.T @ R) - np.ones((N,N)), np.inf))
assert np.linalg.norm(np.exp(1j * N * k.T @ R) - np.ones((N,N)), np.inf) < 1e-8

U = np.exp(1j * k.T @ R) / np.sqrt(N)

I_U = U.conj().T @ U
#plt.imshow(np.abs(I_U))
print(np.linalg.norm(I_U-np.eye(N), np.inf))



#plt.show()














