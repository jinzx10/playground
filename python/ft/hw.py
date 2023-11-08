import numpy as np

L = 20
n = 200
dx = L/n
x = -L/2 + np.arange(n) * dx

mu = 2
sigma = 0.33
f = np.exp(-0.5*((x-mu)/sigma)**2) / (np.sqrt(2*np.pi) * sigma)


from scipy.fft import fft, fftshift

dk = 2*np.pi / (n*dx)
k = ( -(n//2) + np.arange(n) ) * dk
g = dx * fftshift(fft(f)) * np.exp(-1j * k * x[0])

g_ref = np.exp(-0.5*(sigma*k)**2 - 1j*k*mu)
g_diff = np.abs(g-g_ref)
print('max diff = ', np.max(g_diff))


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

fig, ax = plt.subplots(1,3, figsize=(22,7), layout='tight')
ax[0].plot(k, g.real, label='numerical')
ax[0].plot(k, g_ref.real, label='analytical', linestyle='--', linewidth=2)
ax[0].set_title('real part', fontsize=28)
ax[0].set_xlabel('k', fontsize=24)
ax[0].set_ylabel('Re$\\tilde{f}(k)$', fontsize=24)
ax[0].legend(fontsize=20)
ax[0].xaxis.set_tick_params(labelsize=16)
ax[0].yaxis.set_tick_params(labelsize=16)

ax[1].plot(k, g.imag, label='numerical') 
ax[1].plot(k, g_ref.imag, label='analytical', linestyle='--', linewidth=2) 
ax[1].set_title('imaginary part', fontsize=28)
ax[1].set_xlabel('k', fontsize=24)
ax[1].set_ylabel('Im$\\tilde{f}(k)$', fontsize=24)
ax[1].legend(fontsize=20)
ax[1].xaxis.set_tick_params(labelsize=16)
ax[1].yaxis.set_tick_params(labelsize=16)

ax[2].plot(k, np.log10(g_diff+np.finfo(float).eps), linewidth=2)
ax[2].set_title('error', fontsize=28)
ax[2].set_xlabel('k', fontsize=24)
ax[2].set_ylabel('log${}_{10}\\left|\Delta\\tilde{f}(k)\\right|$', fontsize=24)
ax[2].xaxis.set_tick_params(labelsize=16)
ax[2].yaxis.set_tick_params(labelsize=16)


plt.show()

