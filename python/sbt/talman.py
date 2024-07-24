import numpy as np
from scipy.fft import fft, ifft
from scipy.special import loggamma, spherical_jn


def _direct(rho0, drho, f, l, kappa0):
    '''
    Spherical Bessel transform on a log grid via direct discretization.

    This function evaluates the convolution expression of spherical Bessel
    transform via direct discretization on a given log grid. The result is
    used to fix the small k part of the Talman's algorithm. The result is
    not accurate for large k, as the discretization error gets large for a
    strongly oscillating integrand.

    Parameters
    ----------
        rho0 : float
            The initial value of the real-space grid.
        drho : float
            The log spacing of the real-space grid.
            The real-space grid is given by r[i] = exp(rho0 + i*drho).
        f : array_like
            The function to be transformed.
        l : int
            The order of the spherical Bessel transform.
        kappa0 : float
            The initial value of the k-space grid.
            The k-space grid has the same log spacing as the real-space grid.

    '''

    N = len(f)
    r = np.exp( rho0 + np.arange(N) * drho )

    a = np.zeros(2*N)
    a[:N] = f * r**3
    b = spherical_jn(l, np.exp(rho0 + kappa0 + drho * np.arange(2*N)) )

    return drho * fft(fft(a) * ifft(b))[:N].real


def _G_direct(t, l, alpha):
    '''
    Direct evaluation of 

                                         Gamma((l+alpha-it)/2)
    G(t) = \sqrt(pi) * 2^(alpha-2-it) * -----------------------
                                        Gamma((l-alpha+3+it)/2)

    via loggamma function.


    Note
    ----
    This function comes from the integral

        G(t) = \int_0^\infty u^{alpha-1-it} j_l(u) du

    where j_l(u) is the l-th order spherical Bessel function and
    -l < alpha < 2 is a parameter.

    '''
    return np.sqrt(np.pi) * 2**(alpha - 2 - 1j*t) * \
            np.exp( loggamma((l+alpha-1j*t)/2) - loggamma((l-alpha+3+1j*t)/2) )


def _G0(t):
    '''
    Evaluation of _G_direct(t, 0, 1.5) via elementary functions.

    '''
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


def _G1(t):
    '''
    Evaluation of _G_direct(t, 1, 1.5) via elementary functions.

    '''
    phi = np.arctan(np.tanh(0.5*np.pi*t)) - np.arctan(2*t)
    return np.exp(2j*phi) * _G0(t)


def _G_recur(t, l):
    '''
    Evaluation of _G_direct(t, l, 1.5) via a recursive algorithm
    based on elementary functions.

    '''
    if l == 0:
        return _G0(t)
    elif l == 1:
        return _G1(t)
    else:
        return np.exp(-2j * np.arctan( t / (l-0.5) )) * _G_recur(t, l-2)


############################################################
#                           Test
############################################################
import os
import unittest
import matplotlib.pyplot as plt

class _TestTalman(unittest.TestCase):

    def test_talman_direct(self):
        # real-space grid
        rmax = 20
        rmin = 1e-6
        N = 500
        
        rho0 = np.log(rmin)
        drho = np.log(rmax/rmin) / (N-1)
        r = np.exp( rho0 + np.arange(N) * drho )
        
        # k-space grid
        kmin = 1e-6
        kappa0 = np.log(kmin)
        dkappa = drho
        k = np.exp( kappa0 + np.arange(N) * dkappa )
        
        # function to be transformed and reference transformed values
        l = 2
        f = r**2 * np.exp(-r)
        g_ref = 48 * k**2  / (k**2 + 1)**4 

        g = _direct(rho0, drho, f, l, kappa0)

        # check for the small k part
        idx = np.where(k < 1)
        self.assertTrue( np.allclose(g[idx], g_ref[idx], rtol=1e-4, atol=1e-4) )

    
    def test_G(self):
        # cross check _G_direct with _G_recur
        t = np.linspace(-50, 50, 2001)
        lmax = 20
        for l in range(lmax+1):
            g_direct = _G_direct(t, l, 1.5)
            g_recur = _G_recur(t, l)
            self.assertTrue( np.allclose(g_direct, g_recur, rtol=1e-8, atol=1e-8) )

if __name__ == '__main__':
    unittest.main()

