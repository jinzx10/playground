import numpy as np
from sympy.physics.wigner import wigner_d
from sympy import matrix2numpy, Ynm, Matrix

# rotation matrices (counter-clockwise)
def Rz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def Ry(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def R(alpha, beta, gamma):
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)

# rotate [1,0,0] around z-axis by 90 degrees to [0,1,0]
assert np.allclose(Rz(np.pi/2) @ np.array([1, 0, 0]), [0, 1, 0])

# rotate [1,0,0] around y-axis by 90 degrees to [0,0,-1]
assert np.allclose(Ry(np.pi/2) @ np.array([1, 0, 0]), [0, 0, -1])

# R is orthogonal
R_tmp = R(np.random.rand() * 2 * np.pi, np.random.rand() * np.pi, np.random.rand() * 2 * np.pi)
assert np.allclose(R_tmp.T @ R_tmp, np.eye(3))

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    polar = np.arccos(z/r)
    azimuth = np.arctan2(y, x)
    return r, polar, azimuth

def sph2cart(r, polar, azimuth):
    x = r * np.sin(polar) * np.cos(azimuth)
    y = r * np.sin(polar) * np.sin(azimuth)
    z = r * np.cos(polar)
    return x, y, z

# some random vector
u = np.random.randn(3)
r, polar, azimuth = cart2sph(*u)
assert np.allclose(u, sph2cart(r, polar, azimuth))

# Euler angles
alpha = np.random.rand() * 2 * np.pi
beta = np.random.rand() * np.pi
gamma = np.random.rand() * 2 * np.pi

# rotate u
v = R(alpha, beta, gamma) @ u

# spherical harmonics
# NOTE: sympy's Wigner D-matrix arranges m in descending order!!!
# See its source code for details.
def Y(l, polar, azimuth):
    return matrix2numpy(Matrix([Ynm(l, m, polar, azimuth) for m in range(l,-l-1,-1)]), dtype=complex).reshape(-1)

l = 3

r, polar, azimuth = cart2sph(*u)
Yu = Y(l, polar, azimuth)

# Wigner D-matrix
D = matrix2numpy(wigner_d(l, alpha, beta, gamma), dtype=complex)

# inverse
Dinv = matrix2numpy(wigner_d(l, -gamma, -beta, -alpha), dtype=complex)
assert np.allclose(Dinv, np.linalg.inv(D))

# conjugate symmetry
D_conj = matrix2numpy(wigner_d(l, -alpha, beta, -gamma), dtype=complex)
assert np.allclose(D_conj, D.conj())

# NOTE sympy's Wigner D-matrix is defined as the matrix elements of
# exp(i * Jz * alpha) * exp(i * Jy * beta) * exp(i * Jz * gamma)
# which corresponds to the rotation matrix R(-alpha, -beta, -gamma)

v = R(-alpha, -beta, -gamma).T @ u
#v = R(gamma, beta, alpha) @ u # same as above

r, polar, azimuth = cart2sph(*v)
Yv = Y(l, polar, azimuth)

assert np.allclose(Yu @ D, Yv)

