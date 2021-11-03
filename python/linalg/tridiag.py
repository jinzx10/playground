import numpy as np
import matplotlib.pyplot as plt

sz = 10
N = sz-1

alpha = 1
beta = 1.2

H = np.diag(np.ones(sz)*alpha, 0) + np.diag(np.ones(sz-1)*beta, 1) + np.diag(np.ones(sz-1)*beta, -1)

omega = 5.1

t0 = beta / (omega-alpha)
G = np.linalg.inv(omega*np.eye(sz)-H)

k = (1./2/t0) + np.sqrt((1./2/t0)**2-1)
c = (1./2/t0) - np.sqrt((1./2/t0)**2-1)

if abs(t0) > 0.5:
    k = (1./2/t0) + np.sqrt((1./2/t0)**2-1+0j)
    c = (1./2/t0) - np.sqrt((1./2/t0)**2-1+0j)


Delta = G[1:,0] - c * G[:-1,0]


print('G[:,0] = ', G[:,0])

print('ratio = ', G[1:,0] / G[:-1,0])

print('alpha=', alpha)
print('beta=', beta)
print('omega=', omega)
print('t0=', t0)
print('k=', k)

#M = np.diag(np.ones(sz)/t0, 0) - np.diag(np.ones(sz-1), 1) - np.diag(np.ones(sz-1), -1)
#b = np.zeros(sz)
#b[0] = 1./beta
#x = np.linalg.solve(M, b)
#print('x=', x)


n = np.arange(1,N+1,1)
tmp = (1./beta/k**(n-1)) * ( (1-k**(2*N+2))/(1-k**(2*N+4))*(1-k**(2*n+2))/(1-k**2) - (1-k**(2*n))/(1-k**2)   ) 
print(tmp[1:] / tmp[:-1])
print((1./beta/k**(n-1)) * ( (1-k**(2*N+2))/(1-k**(2*N+4))*(1-k**(2*n+2))/(1-k**2) - (1-k**(2*n))/(1-k**2)   )   )
print(1./k**(n+1)/beta)
