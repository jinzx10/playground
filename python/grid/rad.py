import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre, roots_chebyu

import matplotlib.pyplot as plt


# integrate x**2 * f(x) from 0 to inf
def gauss_laguerre(f, n, R):
    # w = theta[i]/( (n+1) * L_{n+1}(theta[i]) )**2
    theta, w = roots_laguerre(n)
    x = theta * R
    w = R**3 * theta**2 * np.exp(theta) * w
    return np.sum(w * f(x))

def becke(f, n, R):
    x, w = roots_chebyu(n)
    w = 2*R**3 * w * (1+x)**1.5 / (1-x)**4.5
    x = (1+x) / (1-x) * R
    return np.sum(w * f(x))





def g(x):
    return np.exp(-x**2)

val_lag = []
val_bec = []

R_list = np.linspace(0.5, 5, 100)
n = 30

val_ref = np.sqrt(np.pi)/4

for R in R_list:
    val_lag.append(gauss_laguerre(g, n, R))
    val_bec.append(becke(g, n, R))

plt.plot(R_list, np.abs(val_lag - val_ref), label='gauss-laguerre')
plt.plot(R_list, np.abs(val_bec - val_ref), label='becke')
plt.legend()
plt.show()

exit(1)

R = 0.7

n = 50
theta, w = roots_laguerre(n)
# w = theta[i]/( (n+1) * L_{n+1}(theta[i]) )**2

x = theta * R
w = R**3 * theta**2 * np.exp(theta) * w

print(np.sum(w * g(x)))
print(0.25 * np.sqrt(np.pi))



exit(1)

def delley(r0, n, mult):

    ngrid = (1+n) * mult - 1
    r = np.zeros(ngrid)

    fac = r0 / np.log(1 - (n/(n+1))**2)
    for i in range(1, ngrid+1):
        r[i-1] = fac * np.log(1-(i/(ngrid+1))**2)

    return r



r0 = 5
n = 30

r = delley(r0, n, 1)
ngrid = len(r)
plt.plot(r, np.zeros(ngrid), '|')
plt.show()

