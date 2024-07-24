import numpy as np

n = 2000
r = np.linspace(-10, 10, n)
dr = r[1] - r[0]

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

omega = 3.0
V = 0.5 * omega**2 * np.diag(r**2)
H = -0.5 * D2r + V
val, vec = np.linalg.eigh(H)
print(val[:5])
