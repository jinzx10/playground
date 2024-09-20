import numpy as np
import matplotlib.pyplot as plt

def delley(r0, n, mult):

    ngrid = (1+n) * mult - 1
    r = np.zeros(ngrid)

    fac = r0 / np.log(1 - (n/(n+1))**2)
    for i in range(1, ngrid+1):
        r[i-1] = fac * np.log(1-(i/(ngrid+1))**2)

    return r


r0 = 5
n = 30

r = delley(r0, n, 2)
ngrid = len(r)
plt.plot(r, np.zeros(ngrid), '|')
plt.show()

