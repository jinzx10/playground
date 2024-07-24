import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.special import loggamma, spherical_jn

import matplotlib.pyplot as plt

class SBT:

    def __init__(self):

        self.lmax = 20

        self.N = 0
        self.drho = 0
        self.rho0_plus_kappa0 = 0

        self.log_grid = None

        self.pre_mult = None
        self.post_mult = None
        self.mult_table1 = None
        self.mult_table2 = None


    def tabulate(self):

        N = self.N
        drho = self.drho
        N_dble = 2 * N

        self.log_grid = np.exp(np.arange(-N, N) * drho)

        # pre-multiplied (r/r0)**1.5 (with extrapolated r grid)
        #self.pre_mult = np.exp(np.arange(-N, N) * drho * 1.5)
        self.pre_mult = self.log_grid**1.5

        # post-multiplied (k/k0)**(-1.5)
        self.post_mult = np.flip(self.pre_mult[1:N+1])

        # intermediate t grid (FFT counter-grid of rho, halved!)
        t = 2 * np.pi / (N_dble * drho) * np.arange(N)

        # mult_table1[l] is Gl times exp[1j*(rho0+kappa0)*t]
        self.mult_table1 = np.zeros((self.lmax+1, N), dtype=complex)
        phi0 = self.rho0_plus_kappa0 * t

        # calculate G0 ~ exp(1j*(phi1+phi2)) where phi1, phi2 are the phases of Gamma(0.5-it) and sin(0.5-it)
        z = 10.5 - 1j*t
        rad = np.sqrt(10.5**2 + t**2)
        theta = np.arctan(t/10.5)

        phi1 = -t * np.log(rad) - 10*theta + t \
                + np.sin(theta)/(12*rad) - np.sin(3*theta)/(360*rad**3) \
                + np.sin(5*theta)/(1260*rad**5) - np.sin(7*theta)/(1680*rad**7)

        for p in range(10):
            phi1 += np.arctan(t/(p+0.5))

        phi2 = -np.arctan(np.tanh(0.5*np.pi*t))

        self.mult_table1[0] = np.sqrt(0.5*np.pi) / N * np.exp(1j*(phi0+phi1+phi2))
        self.mult_table1[0,0] *= 0.5 # NOTE why?

        # calculate G1
        self.mult_table1[1] = np.exp( -2j * (phi2 + np.arctan(2*t)) ) * self.mult_table1[0]

        # use recurrence formula to calculate Gl
        for l in range(2, self.lmax+1):
            self.mult_table1[l] = self.mult_table1[l-2] * np.exp( -2j * np.arctan(t/(l-0.5)) ) 

        # mult_table2 is iFFT of spherical Bessel functions
        self.mult_table2 = np.zeros((self.lmax+1, N_dble), dtype=complex)
        for l in range(self.lmax+1):
            self.mult_table2[l] = ifft(spherical_jn(l, np.exp(self.rho0_plus_kappa0 + N * drho) * self.log_grid ))
        self.mult_table2 *= N_dble


    def talman(self, rho0, drho, f, l, kappa0, pr=0):

        assert( l <= self.lmax )

        N = len(f)
        make_tab = drho != self.drho or N != self.N or rho0 + kappa0 != self.rho0_plus_kappa0

        self.N = N
        self.drho = drho
        self.rho0_plus_kappa0 = rho0 + kappa0

        # initializes variables & tables
        if make_tab:
            self.tabulate()

        # extrapolation
        f_dble = np.zeros(2*N)
        f_dble[:N] = f[0] * self.log_grid[:N]**(pr+l)
        f_dble[N:] = f

        # main algorithm
        tmp = np.zeros(2*N, dtype=complex)
        tmp[:N] = self.mult_table1[l] * ifft(self.pre_mult * f_dble)[:N] * 2 * N
        g = 2 * N * np.exp(1.5 * (rho0 - kappa0)) * ifft(tmp)[N:].real * self.post_mult
        #g = 2 * N * np.exp(1.5 * (rho0 - kappa0)) * fft(tmp.conj())[N:].real * self.post_mult

        # direct discretization
        # tmp is f * r**3 (tmp[N:] remains 0)
        tmp[:N] = f * np.exp(3 * rho0) * self.log_grid[N:]**3
        g_direct = drho / (2*N) * fft( fft(tmp) * self.mult_table2[l] )[:N].real

        # find the transition point
        idx = np.argmin(np.abs(g-g_direct))
        #print('trans point idx = %i,   log10(k)=%8.5f'%(idx, np.log10(np.exp(kappa0+idx*drho))))
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


#def talman(rho0, drho, f, l, kappa0, pr=0):
#    N = len(f)
#
#    # the original r grid follows r = np.exp( rho0 + np.arange(N) * drho )
#    # following Talman, here we double the r grid and extrapolate f as C * r**(pr+l)
#    r_dble = np.exp( rho0 + np.arange(-N, N) * drho )
#    k_dble = np.exp( kappa0 + np.arange(-N, N) * drho )
#    C = f[0] * np.exp(-rho0*(pr+l))
#    f_dble = np.zeros(2*N)
#    f_dble[:N] = C * r_dble[:N]**(pr+l)
#    f_dble[N:] = f
#
#    # intermediate t grid
#    t = np.arange(-N, N) * 2 * np.pi / (2 * N * drho)
#
#    #G = G_direct(t, l, 1.5)
#    G = G_recur(t, l)
#    
#    # transformed array
#    g = 8 * N * np.real(ifft( np.exp(1j*(rho0+kappa0)*t) * ifft(f_dble * r_dble**1.5) * G )) / k_dble**1.5
#    g = g[N:]
#
#    # fix the small k part by using a direct discretization scheme
#    g_direct = talman_direct(rho0, drho, f, l, kappa0)
#
#    # the transition point is chosen to be the point
#    # where the difference between the two methods is smallest
#    idx = np.argmin(np.abs(g-g_direct))
#    print('trans k = ', np.exp(kappa0+drho*idx))
#    g[:idx] = g_direct[:idx]
#
#    return g

def talman(rho0, drho, f, l, kappa0, pr=0):
    N = len(f)

    # the original r grid follows r = np.exp( rho0 + np.arange(N) * drho )

    # intermediate t grid
    t = np.arange(0, N//2) * 2 * np.pi / (N * drho)

    #G = G_direct(t, l, 1.5)
    G = G_recur(t, l)
    
    # transformed array
    g = 8 * N * np.real(ifft( np.exp(1j*(rho0+kappa0)*t) * ifft(f_dble * r_dble**1.5) * G )) / k_dble**1.5
    #g = g[N:]

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


r_dble = np.exp( rho0 + np.arange(-N, N) * drho )
f_dble = np.zeros(2*N)
f_dble[:N] = f[0] * np.arange(-N,0)**l
f_dble[N:] = f

tmp1 = N * ifft(f * r**1.5).real
tmp2 = 2*N * ifft(f_dble * r_dble**1.5)[:N].real

dt = 2 * np.pi / (N * drho)

t1 = np.arange(N) * dt
t2 = np.arange(N) * dt/2
#plt.plot(t1, tmp1)
#plt.plot(t2, tmp2)
#plt.show()

# \utilde{F} from (-N/2)*dt to (N/2-1)*dt
# NOTE: for fftshift would rearrange freqs like -2, -1, 0, 1 instead of -1, 0, 1, 2
uF1_ = N * fftshift(ifft(f * r**1.5))

t1_ = np.arange(-N//2, N//2) * dt
uF1_ *= drho * np.exp(1j * rho0 * t1_)

G1_ = G_recur(t1_, l)

tmp = dt * N * ifft(uF1_ * G1_ * np.exp(1j * kappa0 * t1_)) * (-1) ** np.arange(N)
tmp = tmp.real

tmp /= (2. * np.pi * k**1.5)


plt.plot(k, tmp)
plt.plot(k, g_ref, 'o')
plt.show()


t2_ = np.arange(0, N) * dt/2
uF2 = 2*N * ifft(f_dble * r_dble**1.5)[:N]
#uF2 *= drho * np.exp(1j * rho0 * t2_)

#G2 = G_recur(t2, l)

#plt.plot(t1_, uF1.real)
#plt.plot(t2_, uF2.real)
#plt.show()




exit()


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
#g = talman(rho0, drho, f, l, kappa0)

print('max diff = %8.5e'%(np.max(np.abs(g-g_ref))))

#plt.plot(k, g, linewidth=2, linestyle='-', label='talman')
#plt.plot(k, g_ref, linewidth=2, linestyle='--', label='ref')
#plt.xlim([0.0, k[-1]])
#plt.ylim([-0.5, 4.5])
plt.plot(np.log10(k), np.log10(np.abs(g-g_ref)), linewidth=2, linestyle='-', label='err')
plt.legend(fontsize=20)
plt.show()


