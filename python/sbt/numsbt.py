import numpy as np
from scipy.fft import fft, ifft
from scipy.special import loggamma, spherical_jn

import matplotlib.pyplot as plt

class SBT:

    def __init__(self):

        self.lmax = 20

        self.rho0 = None
        self.drho = None
        self.kappa0 = None
        self.N = None

        self.r0 = None
        self.k0 = None

        self.r = None
        self.k = None

        self.r_small = None

        self.premult = None
        self.postdiv = None
        self.mult_table1 = None
        self.mult_table2 = None


    def prelude(self, rho0, drho, kappa0, N):

        self.rho0 = rho0
        self.drho = drho
        self.kappa0 = kappa0
        self.N = N

        self.r0 = np.exp(rho0)
        self.k0 = np.exp(kappa0)

        self.r = np.exp(rho0 + np.arange(N) * drho)
        self.k = np.exp(kappa0 + np.arange(N) * drho)

        N_dble = 2 * N
        dt = 2 * np.pi / (N_dble * drho)

        # extrapolated r grid
        self.r_small = np.exp(rho0 + np.arange(-N, 0) * drho)

        # pre-multiplied (r/r0)**1.5
        self.premult = np.zeros(N_dble)
        self.premult[N:] = np.exp(np.arange(N) * drho * 1.5)
        self.premult[:N] = np.exp(np.arange(-N, 0) * drho * 1.5)

        self.postdiv = np.exp(np.arange(0, -N, -1) * drho * 1.5)

        # intermediate t grid
        t = np.arange(N) * dt

        # mult_table1 is Gn times exp[i(rho0+kappa0)*tn]
        self.mult_table1 = np.zeros((self.lmax+1, N), dtype=complex)
        phi0 = (rho0 + kappa0) * t

        z = 10 + 0.5 - 1j*t
        rad = np.sqrt(10.5**2 + t**2)
        theta = np.arctan(t/10.5)

        phi1 = -t * np.log(rad) - 10*theta + t \
                + np.sin(theta)/(12*rad) - np.sin(3*theta)/(360*rad**3) \
                + np.sin(5*theta)/(1260*rad**5) - np.sin(7*theta)/(1680*rad**7)

        for p in range(10):
            phi1 += np.arctan(t/(p+0.5))

        phi2 = -np.arctan(np.tanh(0.5*np.pi*t))

        self.mult_table1[0] = np.sqrt(0.5*np.pi) / N * np.exp(1j*(phi0+phi1+phi2))
        self.mult_table1[0,0] *= 0.5 # why?

        self.mult_table1[1] = np.exp( 2j * (-phi2 - np.arctan(2*t)) ) * self.mult_table1[0]

        for l in range(2, self.lmax+1):
            self.mult_table1[l] = self.mult_table1[l-2] * np.exp( -2j * np.arctan(t/(l-0.5)) ) 

        #print(self.mult_table1[2]) # pass

        # table of spherical bessel functions
        jl_table = np.zeros((self.lmax+1, N_dble))
        for l in range(self.lmax+1):
            jl_table[l] = spherical_jn(l, np.exp(self.rho0 + self.kappa0 + np.arange(N_dble) * drho))

        #print(jl_table[20]) # pass

        # mult_table2 is FFT of jl_table, conjugated
        self.mult_table2 = np.zeros((self.lmax+1, N_dble), dtype=complex)
        for l in range(self.lmax+1):
            self.mult_table2[l] = ifft(jl_table[l]).conj() * N_dble

        #print(self.mult_table2[0]) # pass

    def talman(self, rho0, drho, f, l, kappa0, pr=0):

        N = len(f)

        # initializes variables & tables
        self.prelude(rho0, drho, kappa0, N)
        
        # extrapolation
        f_dble = np.zeros(2*N)
        C = f[0] * np.exp(-rho0*(pr+l))
        f_dble[:N] = C * self.r_small**(pr+l)
        f_dble[N:] = f

        tmp2 = 2 * self.N * ifft(self.premult * f_dble)
        tmp1 = np.zeros(2*N, dtype=complex)
        tmp1[:N] = self.mult_table1[l] * tmp2[:N]

        tmp2 = 2 * N * ifft(tmp1)

        factor = (self.r0 / self.k0)**1.5
        g = factor * tmp2[N:].real * self.postdiv

        #print(g)

        # direct discretization
        tmp3 = np.zeros(2*N)
        tmp3[:N] = f * self.r**3

        g_direct = 2 * N * drho * ifft( ifft(tmp3) * self.mult_table2[l] )[:N].real
        print(g_direct)

        # find transition point
        idx = np.argmin(np.abs(g-g_direct))
        #print('trans point idx = ', idx)
        g[:idx] = g_direct[:idx]

        return g



#################################################
#
#   recursive algorithm for G below assumes
#
#           alpha = 1.5
#
#################################################
def G0(t):

    # This function implements Eq. 19 of Talman2009

    N = 10 
    z = N + 0.5 - 1j*t
    r = np.abs(z)
    theta = np.arctan(t/(N+0.5))

    # phi1: phase of Gamma(1/2 - it)
    phi1 = -t * np.log(r) - N*theta + t \
            + np.sin(theta)/(12*r) - np.sin(3*theta)/(360*r**3) \
            + np.sin(5*theta)/(1260*r**5) - np.sin(7*theta)/(1680*r**7)
    for p in range(N):
        phi1 += np.arctan(t/(p+0.5))

    # phi2: phase of sin(1/2 - it)
    phi2 = -np.arctan(np.tanh(0.5*np.pi*t))

    return np.sqrt(0.5*np.pi) * np.exp(1j*(phi1+phi2))


def G1(t):
    # This function implements Eq. 16 of Talman2009
    phi = np.arctan(np.tanh(0.5*np.pi*t)) - np.arctan(2*t)
    return np.exp(2j*phi) * G0(t)


def G_recur(t, l):
    if l == 0:
        return G0(t)
    elif l == 1:
        return G1(t)
    else:
        # Eq. 24 of Talman2009
        return np.exp(-2j*np.arctan(t/(l-0.5))) * G_recur(t, l-2)

#################################################
#
#   direct calculation of G using loggamma
#
#################################################
def G_direct(t, l, alpha):
    return np.sqrt(np.pi) * 2**(alpha-2-1j*t) * np.exp( loggamma((l+alpha-1j*t)/2) - loggamma((l-alpha+3+1j*t)/2) )


#################################################
#
#               main program
#
#################################################
def talman_direct(rho0, drho, f, l, kappa0):

    # Direct discretization of the convolution (Eqs. 34-38 of Talman2009)

    N = len(f)
    r = np.exp( rho0 + np.arange(N) * drho )

    a = np.zeros(2*N)
    a[:N] = f * r**3
    b = spherical_jn(l, np.exp(rho0+kappa0+drho*np.arange(2*N)) )

    g_direct = drho / (2*N) * fft( fft(a) * fft(b).conj() ).real
    return g_direct[:N]


def talman(rho0, drho, f, l, kappa0, pr=0):
    N = len(f)

    # the original r grid follows r = np.exp( rho0 + np.arange(N) * drho )
    # following Talman, here we double the r grid and extrapolate f as C * r**(pr+l)
    r_dble = np.exp( rho0 + np.arange(-N, N) * drho )
    k_dble = np.exp( kappa0 + np.arange(-N, N) * drho )
    C = f[0] * np.exp(-rho0*(pr+l))
    f_dble = np.zeros(2*N)
    f_dble[:N] = C * r_dble[:N]**(pr+l)
    f_dble[N:] = f

    # intermediate t grid
    t = np.arange(2 * N) * 2 * np.pi / (2 * N * drho)

    #G = G_direct(t, l, 1.5)
    G = G_recur(t, l)
    
    # transformed array
    g = 8 * N * np.real(ifft( np.exp(1j*(rho0+kappa0)*t) * ifft(f_dble * r_dble**1.5) * G )) / k_dble**1.5
    g = g[N:]

    # fix the small k part by using a direct discretization scheme
    g_direct = talman_direct(rho0, drho, f, l, kappa0)

    # the transition point is chosen to be the point
    # where the difference between the two methods is smallest
    idx = np.argmin(np.abs(g-g_direct))
    print('trans k = ', np.exp(kappa0+drho*idx))
    g[:idx] = g_direct[:idx]

    return g

#################################################
#
#                   test
#
#################################################
alpha = 1.5
N = 300

# real-space log grid
rmax = 30
rmin = 1e-4

rho0 = np.log(rmin)
drho = np.log(rmax/rmin) / (N-1)
r = np.exp( rho0 + np.arange(N) * drho )

# k-space log grid
kmax = 500
kappa0 = np.log( kmax / (rmax/rmin) )
dkappa = drho
k = np.exp( kappa0 + np.arange(N) * dkappa )

# SBT pair
# Talman's SBT differs by a factor of sqrt(2/pi)
l = 2
f = r**2 * np.exp(-r)
g_ref = 48 * k**2  / (k**2 + 1)**4 

# Talman2009 Eq. 40
#l = 0
#lam = 1
#f = np.exp(-lam*r)
#g_ref = 2 * lam / (lam**2 + k**2)**2


#plt.plot(r, f)
#plt.plot(k, g_ref)

# intermediate t grid
#dt = 2 * np.pi / (N * drho)
#t = np.arange(N) * dt
#
## \tilde{G}
##G = G_direct(t, l, alpha)
#G = G_recur(t, l)
#
#g = 2 * N * np.real(ifft( np.exp(1j*(rho0+kappa0)*t) * ifft(f * r**(3-alpha)) * G )) / k**(alpha)

sbt = SBT()
g = sbt.talman(rho0, drho, f, l, kappa0)

print('max diff = %8.5e'%(np.max(np.abs(g-g_ref))))

#plt.plot(k, g, linewidth=2, linestyle='-', label='talman')
#plt.plot(k, g_ref, linewidth=2, linestyle='--', label='ref')
#plt.xlim([0.0, k[-1]])
#plt.ylim([-0.5, 4.5])
plt.plot(np.log10(k), np.log10(np.abs(g-g_ref)), linewidth=2, linestyle='-', label='err')
plt.legend(fontsize=20)
plt.show()


