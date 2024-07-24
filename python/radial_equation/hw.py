import numpy as np

n = 2000
r = np.linspace(-20, 20, n)
dr = r[1] - r[0]

# d^2/dr^2
D2r = np.zeros((n, n))
for i in range(1, n-1):
    D2r[i, i-1] = 1
    D2r[i, i] = -2
    D2r[i, i+1] = 1
D2r[0, 0] = -2
D2r[0, 1] = 1
D2r[-1, -1] = -2
D2r[-1, -2] = 1
D2r /= dr**2

# d/dr
Dr = np.zeros((n, n))
for i in range(1, n-1):
    Dr[i, i-1] = -1
    Dr[i, i+1] = 1
Dr[0, 1] = 1
Dr[-1, -2] = -1
Dr /= 2*dr

# coulomb
V = np.diag(-1/r)

# hamiltonian
l = 1
#H = -D2r + np.diag(2.0/r) @ Dr - np.diag(2.0/r**2) + 2.0 * V + l*(l+1)*np.diag(1/r**2)

H = D2r + np.diag(2.0/r) @ Dr + np.diag(1.0/r) - l*(l+1)*np.diag(1/r**2)


val, vec = np.linalg.eigh(H)
val = np.sort(val)[::-1]
print(np.sqrt(val[:5]))

