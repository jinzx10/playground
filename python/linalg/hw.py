import numpy as np

# random eigenvalues summed to 1
eigval = np.random.rand(10)
eigval = eigval / np.sum(eigval)

# random orthogonal matrix
Q,_ = np.linalg.qr(np.random.randn(10,10))

rho = Q @ np.diag(eigval,0) @ Q.T

# reduced density matrix
# |00>, |01>, |10>, |11>
sigma = np.zeros((4,4))

# <00| |00>
sigma[0,0] = rho[0,0]

#
sigma[1,1] = rho[1,1] + rho[2,2] + rho[3,3]
sigma[1,2] = rho[1,4] + rho[2,5] + rho[3,6]
sigma[2,1] = rho[4,1] + rho[5,2] + rho[6,3]
sigma[2,2] = rho[4,4] + rho[5,5] + rho[6,6]

# <11| |11>
sigma[3,3] = rho[7,7] + rho[8,8] + rho[9,9]

print(sigma)

val, vec = np.linalg.eigh(sigma)

print('val = ', val)
print('sum(val) = ', np.sum(val))
print('vec = ', vec)
