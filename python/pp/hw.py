import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.integrate import simpson

# box [0, L)
L = 10

'''
          -
          |  1 + cos(pi*(x-x0)/w)      x0-w <= x <= x0+w
beta(x) = |  
          |  0                          otherwise
          -

'''
w = 1.11
x0 = 2.34 # x0 has to be larger than w
def beta(grid):
    val = 1.0 + np.cos(np.pi * (grid - x0) / w)
    mask = (grid >= x0-w) & (grid <= x0+w)
    return val * mask
    
'''

psi = sum_G c_G exp(iGx)

G = 2*k*pi/L where k = -m, ..., 0, 1, ..., m

'''
m = 10
n = 2*m + 1
G = 2*np.pi/L * (np.arange(n) - n//2)
c = np.random.randn(len(G)) + 1j * np.random.randn(len(G))
def psi(grid):
    return c @ np.exp(1j * np.outer(G, grid))


#x_plt = np.linspace(0, L, 200)
#beta_plt = beta(x_plt)
#psi_plt = psi(x_plt)
#plt.plot(x_plt, np.real(psi_plt))
#plt.plot(x_plt, beta_plt)
#plt.xlim([0, L])
#plt.show()

# <beta|psi>

# analytic
# <beta|G>
def beta_g(g):
    a = g * w
    return w * np.exp(1j * g * x0) * (np.sinc(a/np.pi) + a*np.sin(a)/(np.pi**2-a**2)) * 2

#beta_G = w * np.exp(1j * G * x0) * (np.sinc(a/np.pi) + a*np.sin(a)/(np.pi**2-a**2)) * 2
beta_G = beta_g(G)
ref = c @ beta_G

# numerical (brute force)
x = np.linspace(x0-w, x0+w, 1000)
val = simpson(beta(x) * psi(x), x)
print(f'val = {val}')
print(f'ref = {ref}')

#G2 = 2*np.pi/L * (np.arange(4*n+1) - 2*n)
#G4 = 2*np.pi/L * (np.arange(8*n+1) - 4*n)
#
#print(beta_g(G))
#print(beta_g(G2))
#print(beta_g(G4))

# crude evaluation
dx = L / n
x = dx * np.arange(n)
tmp = dx * np.sum(beta(x) * psi(x))
print(f'tmp = {tmp}')



