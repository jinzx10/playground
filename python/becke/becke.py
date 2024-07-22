import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(-1, 1, 100)

def p(x):
    return 1.5 * x - 0.5 * x**3

#fig, ax = plt.subplots(1, 2, layout='tight', figsize=(8,3))
fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))

plt.subplots_adjust(left=0.10, bottom=0.18, right=0.98, top=0.9, wspace=0.3)
g = np.copy(mu)
for i in range(1, 6):
    g = p(g)
    linestyle = '-' if i == 3 else ':'

    if i == 3:
        s_becke = 0.5*(1-g)

    ax[0].plot(mu, g, label=f'$p_{i}$', linestyle=linestyle, linewidth=1)

    ax[0].set_xlim([-1,1])
    ax[0].set_ylim([-1,1.01])

    ax[0].legend()
    ax[0].set_xlabel('$\mu$')
    ax[0].set_ylabel('$g(\mu)$')
    ax[0].set_title('Becke\'s iterated polynomial')


# Stratmann
a = 0.64

def z_func(x):
    return ( 35*(x/a) - 35*(x/a)**3 + 21*(x/a)**5 - 5*(x/a)**7 ) / 16

def g_func(x):
    y = np.zeros_like(x)
    y[x <= -a] = -1
    y[x >= a] = 1
    idx = np.logical_and(x > -a, x < a)
    y[idx] = z_func(x[idx])
    return y

s_stratmann = 0.5 * (1 - g_func(mu))

ax[1].plot(mu, s_stratmann, label='Stratmann', linestyle='-', linewidth=1, color='C0')
ax[1].plot(mu, s_becke, label='Becke', linestyle=':', linewidth=1, color='red')

ax[1].set_xlim([-1, 1])
ax[1].set_ylim([0, 1.005])

ax[1].legend()
ax[1].set_xlabel('$\mu$')
ax[1].set_ylabel('$s(\mu)$')
ax[1].set_title('Becke vs. Stratmann')

plt.savefig('/mnt/c/Users/jzx01/Downloads/becke.png', dpi=600)
#plt.show()



