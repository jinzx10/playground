import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre, roots_chebyu

import matplotlib.pyplot as plt


'''
    
    / inf
    |    x**2 * f(x) dx
    / 0


'''

def gauss_laguerre(f, n, R):
    theta, w = roots_laguerre(n)
    # w = theta[i]/( (n+1) * L_{n+1}(theta[i]) )**2

    x = theta * R
    w = R**3 * theta**2 * np.exp(theta) * w
    return np.sum(w * f(x))

def becke(f, n, R):
    x, w = roots_chebyu(n)
    # standard Chebyshev-Gauss quadrature yields w = pi/(n+1) * (1-x**2)

    w = 2 * np.pi * R**3 / (n+1) * (1+x)**2.5 / (1-x)**3.5
    x = (1+x) / (1-x) * R
    return np.sum(w * f(x))

def murray(f, n, R):
    x = np.arange(1, n+1) / (n+1)
    w = 2*R**3 * x**5 / (1-x)**7 / (n+1)

    x = (x/(1-x))**2 * R
    return np.sum(w * f(x))

def baker(f, n, R):
    i = np.arange(1, n+1)
    r = R * np.log(1-(i/(n+1))**2) / np.log(1-(n/(n+1))**2)
    w = -2 * i * R * r**2 / np.log(1-(n/(n+1))**2) / ((n+1)**2-i**2)
    return np.sum(w * f(r))

def ta3(f, n, R, alpha):
    x = np.cos(np.arange(1, n+1) * np.pi / (n+1))
    w = np.pi / (n+1) * R**3 * (1+x)**(3*alpha) / np.log(2)**3 * (np.sqrt((1+x)/(1-x))*np.log((1-x)/2)**2 \
            -alpha * np.sqrt((1-x)/(1+x)) * np.log((1-x)/2)**3)
    x = -R * (1+x)**alpha / np.log(2) * np.log((1-x)/2)
    return np.sum(w * f(x))

def knowles(f, n, R):
    x = np.arange(1, n+1) / (n+1)
    w = 3 * x**2 * np.log(1-x**3)**2 / (1-x**3) / (n+1) * R**3
    x = -R * np.log(1-x**3)
    return np.sum(w * f(x))

def de2(f, n, R, alpha):
    h = 0.3
    x = np.arange(1, n+1) * h
    w = h * np.exp(3*alpha*x-3*np.exp(-x)) * (alpha + np.exp(-x))
    x = np.exp(alpha*x - np.exp(-x))
    return np.sum(w * f(x))


def g(x):
    return np.exp(-x**2)

val_ref = np.sqrt(np.pi)/4
val_lag = []
val_bec = []
val_mur = []
val_bak = []
val_kno = []
val_ta3 = []
val_de2 = []

Rs = np.linspace(0.1, 10, 100)
n = 30


for R in Rs:
    val_lag.append(gauss_laguerre(g, n, R))
    val_bec.append(becke(g, n, R))
    val_mur.append(murray(g, n, R))
    val_bak.append(baker(g, n, R))
    val_kno.append(knowles(g, n, R))
    val_ta3.append(ta3(g, n, R, 0.6)) # alpha=0.6
    val_de2.append(de2(g, n, 5, R)) # change the scaling parameter, not cutoff

plt.plot(Rs, np.abs(val_lag - val_ref), label='gauss-laguerre')
plt.plot(Rs, np.abs(val_bec - val_ref), label='becke')
plt.plot(Rs, np.abs(val_mur - val_ref), label='murray')
plt.plot(Rs, np.abs(val_bak - val_ref), label='baker')
plt.plot(Rs, np.abs(val_kno - val_ref), label='knowles')
plt.plot(Rs, np.abs(val_ta3 - val_ref), label='ta3')
plt.plot(Rs, np.abs(val_de2 - val_ref), label='de2')

plt.legend()
#plt.xscale('log')
plt.yscale('log')
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

