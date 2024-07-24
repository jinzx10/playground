import numpy as np

#########################################
#       a minimal 1-d example
#########################################
# lattice contant
a = 5

# reciprocal lattice constant
b = 2 * np.pi / a

# lattice points
nR = 9
shift = -3
R = (shift + np.arange(nR)) * a

sz = 3

# H(R)
H_R = np.random.randn(nR, sz, sz)

# k sampling
nk = 33
k = np.linspace(-0.5, 0.5, 33, endpoint=False) * b
dk = k[1] - k[0]
assert np.abs(nk * dk - b) < 1e-14
k += 0.5 * dk # shift k to become a symmetric grid

# H(k)
H_k = np.zeros((nk, sz, sz), dtype=complex)
for ik in range(nk):
    for iR in range(nR):
        H_k[ik] += H_R[iR] * np.exp(-1j * k[ik] * R[iR])

# recover H(R)
H_R2 = np.zeros((nR, sz, sz), dtype=complex)
for iR in range(nR):
    for ik in range(nk):
        H_R2[iR] += H_k[ik] * np.exp(1j * R[iR] * k[ik]) / nk

assert np.all([np.linalg.norm(H_R[iR] - H_R2[iR], np.inf) < 1e-14 for iR in range(nR)])


