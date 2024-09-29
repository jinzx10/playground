import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre, roots_chebyu, laguerre

import matplotlib.pyplot as plt


'''
    
    / inf
    |    r**2 * f(r) dx ~ \sum_i w[i] * f(r[i])
    / 0


'''

def gauss_laguerre(f, n, R):
    theta, w = roots_laguerre(n)
    x = theta * R
    # w = theta[i]/( (n+1) * L_{n+1}(theta[i]) )**2
    #w = x**3 * np.exp(theta) / ((n+1) * laguerre(n+1)(theta))**2
    w = R**3 * theta**2 * np.exp(theta) * w
    return np.sum(w * f(x))


def becke(f, n, R):
    x, w = roots_chebyu(n)
    # standard Chebyshev-Gauss quadrature yields w = pi/(n+1) * (1-x**2)

    w = 2 * np.pi * R**3 / (n+1) * (1+x)**2.5 / (1-x)**3.5
    r = (1+x) / (1-x) * R
    return np.sum(w * f(r))


def murray(f, n, R):
    x = np.arange(1, n+1) / (n+1)
    w = 2*R**3 * x**5 / (1-x)**7 / (n+1)

    r = (x/(1-x))**2 * R
    return np.sum(w * f(r))


def baker(f, n, R):
    i = np.arange(1, n+1)
    r = R * np.log(1-(i/(n+1))**2) / np.log(1-(n/(n+1))**2)
    w = -2 * i * R * r**2 / np.log(1-(n/(n+1))**2) / ((n+1)**2-i**2)
    return np.sum(w * f(r))


def aims(f, n, R, mult):
    ngrid = (n+1) * mult - 1
    fac = R / np.log(1 - (n/(n+1))**2)
    i = np.arange(1, ngrid+1)
    r = fac * np.log(1-(i/(ngrid+1))**2)
    w = -2 * i * fac * r**2 / ((ngrid+1)**2-i**2)
    return np.sum(w * f(r))


def ta3(f, n, R):
    x = np.cos(np.arange(1, n+1) * np.pi / (n+1))
    w = np.pi / (n+1) * (R/np.log(2))**3 * np.sqrt((1+x)/(1-x)) * np.log(2/(1-x))**2
    r = R/np.log(2) * np.log(2/(1-x))
    return np.sum(w * f(r))

def ta4(f, n, R, alpha):
    x = np.cos(np.arange(1, n+1) * np.pi / (n+1))
    w = np.pi / (n+1) * R**3 * (1+x)**(3*alpha) / np.log(2)**3 * (np.sqrt((1+x)/(1-x))*np.log((1-x)/2)**2 \
            -alpha * np.sqrt((1-x)/(1+x)) * np.log((1-x)/2)**3)
    r = -R * (1+x)**alpha / np.log(2) * np.log((1-x)/2)
    return np.sum(w * f(r))


def knowles(f, n, R):
    x = np.arange(1, n+1) / (n+1)
    w = 3 * x**2 * np.log(1-x**3)**2 / (1-x**3) / (n+1) * R**3
    r = -R * np.log(1-x**3)
    return np.sum(w * f(r))


def de2_qchem(f, n, alpha):
    h = 2.0 / (n - 1)
    x = h * np.arange(n) - 1.0
    ax = alpha * x
    emx = np.exp(-x)
    w = h * np.exp(3*(ax-emx)) * (alpha + emx)
    r = np.exp(ax - emx)
    print(f"h = {h}    x[0] = {x[0]}    x[-1] = {x[-1]}    r[0] = {r[0]}    r[-1] = {r[-1]}")
    return np.sum(w * f(r))

def de2(f, n, alpha, rmin=1e-7, rmax=10):
    #xmax = np.log(rmax) / alpha
    #xmin = -np.log(-np.log(rmin))
    #h = (xmax - xmin) / (n - 1)
    #x = xmin + h * np.arange(n)
    h = 5.0 / (n - 1)
    x = h * np.arange(n) - 1.0
    ax = alpha * x
    emx = np.exp(-x)
    w = h * np.exp(3*(ax-emx)) * (alpha + emx)
    r = np.exp(ax - emx)
    #print(f"h = {h}    x[0] = {x[0]}    x[-1] = {x[-1]}    r[0] = {r[0]}    r[-1] = {r[-1]}")
    return np.sum(w * f(r))


def g(x):
    return np.exp(-x**2)


def de_x(x, alpha):
    return np.exp(alpha * x - np.exp(-x))


#x = np.linspace(-1, 1, 20)
#plt.plot(x, np.log(de_x(x, 0.5)), label='0.5')
#plt.plot(x, np.log(de_x(x, 1.0)), label='1.0')
#plt.plot(x, np.log(de_x(x, 2.0)), label='2.0')
#plt.plot(x, np.log(de_x(x, 3.0)), label='3.0')
#plt.plot(x, np.log(de_x(x, 5.0)), label='5.0')
#plt.legend()
#plt.show()
#exit()



val_ref = np.sqrt(np.pi)/4
val_lag = []
val_bec = []
val_mur = []
val_bak = []
val_kno = []
val_ta3 = []
val_ta4 = []
val_de2 = []
val_aims_2 = []
val_aims_4 = []

Rs = np.linspace(0.5, 7, 100)
n = 30


for R in Rs:
    val_lag.append(gauss_laguerre(g, n, 1/R))
    val_bec.append(becke(g, n, R))
    val_mur.append(murray(g, n, R))
    val_bak.append(baker(g, n, R))
    val_kno.append(knowles(g, n, R))
    val_ta3.append(ta3(g, n, R))
    val_ta4.append(ta4(g, n, R, 0.6)) # alpha=0.6
    val_de2.append(de2_qchem(g, n, R)) # the parameter is used as scaling, not cutoff
    val_aims_2.append(aims(g, n, R, 2))
    val_aims_4.append(aims(g, n, R, 4))

plt.plot(Rs, np.abs(val_lag - val_ref), label='gauss-laguerre')
plt.plot(Rs, np.abs(val_bec - val_ref), label='becke')
plt.plot(Rs, np.abs(val_mur - val_ref), label='murray')
plt.plot(Rs, np.abs(val_bak - val_ref), label='baker')
plt.plot(Rs, np.abs(val_kno - val_ref), label='knowles')
plt.plot(Rs, np.abs(val_ta3 - val_ref), label='ta3')
plt.plot(Rs, np.abs(val_ta4 - val_ref), label='ta4')
plt.plot(Rs, np.abs(val_de2 - val_ref), label='de2')
plt.plot(Rs, np.abs(val_aims_2 - val_ref), label='AIMS-2')
plt.plot(Rs, np.abs(val_aims_4 - val_ref), label='AIMS-4')

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

