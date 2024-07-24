import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from jlzeros import bracket_d, ikebe

def Nfac(l, q, rcut):
    '''
    Normalization factor for a truncated spherical Bessel function.

    '''
    r = np.linspace(0, rcut, int(rcut/0.01)+1)
    return 1. / np.sqrt( simpson((r * spherical_jn(l, q*r))**2, r) )

def jlbar(l, q, r, d):
    '''
    d-th derivative of the normalized truncated spherical Bessel function j_l(q*r).

    The function is normalized such that the integral of j_l(q*r)**2 * r**2 over r is 1.

    '''
    rcut = r[-1] if isinstance(r, list) or isinstance(r, np.ndarray) else r
    if d == 0:
        return spherical_jn(l, q*r) * Nfac(l, q, rcut)
    else:
        if l == 0:
            return -Nfac(0, q, rcut) * q * jlbar(1, q, r, d-1) / Nfac(1, q, rcut)
        else:
            return  ( l * jlbar(l-1, q, r, d-1) / Nfac(l-1, q, rcut) \
                    - (l+1) * jlbar(l+1, q, r, d-1) / Nfac(l+1, q, rcut) ) \
                    * Nfac(l, q, rcut) * q / (2 * l + 1)

l = 2
rcut = 9
r = np.linspace(0, rcut, int(rcut/0.01)+1)

nq = 10
zeros_d = bracket_d(l, nq)
zeros_1 = ikebe(l+1, nq)
zeros = ikebe(l, nq)

q = zeros[1]/rcut
f = jlbar(l, q, r, 0)
df = jlbar(l, q, r, 1)
d2f = jlbar(l, q, r, 2)
print(q*q)
print(-2*simpson(r*f*df, r) - simpson(r**2*f*d2f, r) + l*(l+1)*simpson(f**2, r))
exit()


for i in range(nq):
    print(jlbar(l, zeros[i]/rcut, rcut, 1) / jlbar(l, zeros[i]/rcut, rcut, 2))


exit()

# finite difference check below
q0 = zeros[2]/rcut
f0 = jlbar(l, q0, r, 0)

q1 = zeros_d[2]/rcut
f1 = jlbar(l, q1, r, 0)

print('normalization: ', simpson(f0**2 * r**2, r))
print('normalization: ', simpson(f1**2 * r**2, r))

plt.axhline(0, color='black')
#plt.plot(r, f0)
#plt.plot(r, f1)
#plt.show()
#exit()

df0 = jlbar(l, q0, r, 1)
df0_fd = np.diff(f0) / np.diff(r)

df1 = jlbar(l, q1, r, 1)
df1_fd = np.diff(f1) / np.diff(r)

#plt.plot(r, df0)
#plt.plot(r, df1)
r_fd = (r[:-1]+r[1:])/2
#plt.plot(r_fd, df0_fd, linestyle='--')
#plt.plot(r_fd, df1_fd, linestyle='--')
#plt.show()
#exit()

d2f0 = jlbar(l, q0, r, 2)
d2f0_fd = np.diff(df0_fd) / np.diff(r_fd)

d2f1 = jlbar(l, q1, r, 2)
d2f1_fd = np.diff(df1_fd) / np.diff(r_fd)

#plt.plot(r, d2f0)
#plt.plot(r, d2f1)
r_fd = (r_fd[:-1]+r_fd[1:])/2
#plt.plot(r_fd, d2f0_fd, linewidth=4, linestyle=':')
#plt.plot(r_fd, d2f1_fd, linewidth=4, linestyle=':')
#plt.show()
#exit()

exit()



