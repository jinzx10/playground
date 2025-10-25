import numpy as np
import matplotlib.pyplot as plt

M = 200
x = (1 + np.arange(M)) / M
eta = 7


O = np.diag(np.sin(2*eta*x)/(2*x) + M*np.pi - eta)

for i in range(M):
    for j in range(i):
        xp = x[i] + x[j]
        xm = x[i] - x[j]
        O[i,j] = np.sin(xp*eta)/xp - np.sin(xm*eta)/xm
        O[j,i] = O[i,j]

val, vec = np.linalg.eigh(O)

mask = vec[:,0]/x

plt.plot(x, mask)
plt.show()
