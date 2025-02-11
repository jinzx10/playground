import numpy as np

L = 20
n = 100
dx = L/n
x = -L/2 + np.arange(n) * dx

p = 5
f = np.exp(-x*x + 1j*p*x)


from scipy.fft import fft, ifft
dk = 2*np.pi / (n*dx)
k0 = -np.pi/dx - p
k = k0 + np.arange(n) * dk
#g = dx * n * ifft(f * np.exp(1j * k0 * x)) * np.exp(1j * dk * np.arange(n) * x[0])
g = dx * n * ifft(f * np.exp(1j * k0 * dx * np.arange(n))) * np.exp(1j * k * x[0])


g_ref = np.sqrt(np.pi) * np.exp(-0.25*(k+p)**2)
g_diff = np.abs(g-g_ref)
print('max diff = ', np.max(g_diff))


import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

fig, ax = plt.subplots(1,3, figsize=(15,5), layout='tight')
ax[0].plot(k, g.real, label='numerical')
ax[0].plot(k, g_ref.real, label='analytical', linestyle='--', linewidth=2)
ax[0].set_title('real part', fontsize=24)
ax[0].set_xlabel('$k$', fontsize=20)
ax[0].set_ylabel('Re$\\underline{f}(k)$', fontsize=20)
ax[0].legend(fontsize=16)
ax[0].xaxis.set_tick_params(labelsize=16)
ax[0].yaxis.set_tick_params(labelsize=16)
ax[0].set_xlim((k[0], k[-1]))

ax[1].plot(k, g.imag, label='numerical') 
ax[1].plot(k, g_ref.imag, label='analytical', linestyle='--', linewidth=2) 
ax[1].set_title('imaginary part', fontsize=24)
ax[1].set_xlabel('$k$', fontsize=20)
ax[1].set_ylabel('Im$\\underline{f}(k)$', fontsize=20)
ax[1].legend(fontsize=16)
ax[1].xaxis.set_tick_params(labelsize=16)
ax[1].yaxis.set_tick_params(labelsize=16)
ax[1].set_xlim((k[0], k[-1]))

ax[2].plot(k, np.log10(g_diff+np.finfo(float).eps), linewidth=2)
ax[2].set_title('error', fontsize=24)
ax[2].set_xlabel('$k$', fontsize=20)
ax[2].set_ylabel('log${}_{10}\\left|\Delta\\underline{f}(k)\\right|$', fontsize=20)
ax[2].xaxis.set_tick_params(labelsize=16)
ax[2].yaxis.set_tick_params(labelsize=16)
ax[2].set_xlim((k[0], k[-1]))

plt.savefig('gaussian_wave_packet.png', dpi=600)
#plt.show()

