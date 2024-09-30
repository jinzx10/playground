import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_laguerre

import matplotlib.pyplot as plt


'''
Grid and weights for the quadrature of the radial integrals
    
    / inf
    |    r**2 * f(r) dx ~ \sum_i w[i] * f(r[i])
    / 0


'''

def gauss_laguerre(n, R):
    theta, w = roots_laguerre(n)
    r = theta * R
    w = R**3 * theta**2 * np.exp(theta) * w
    return r, w


def becke(n, R):
    x = np.cos(np.arange(1, n+1) * np.pi / (n+1))
    w = 2 * np.pi * R**3 / (n+1) * (1+x)**2.5 / (1-x)**3.5
    r = (1+x) / (1-x) * R
    return r, w


def murray(n, R):
    x = np.arange(1, n+1) / (n+1)
    w = 2*R**3 * x**5 / (1-x)**7 / (n+1)
    r = (x/(1-x))**2 * R
    return r, w


def baker(nbase, Rc, mult):
    n = (nbase+1) * mult - 1
    R = -Rc / np.log((2*nbase+1) / ((nbase+1)*(nbase+1)));
    x = np.arange(1, n+1) / (n+1)
    r = -R * np.log(1-x*x)
    w = 2 * R * r**2 * x / (1-x**2) / (n+1)
    return r, w


def treutler_m4(n, R, alpha=0.6):
    x = np.cos(np.arange(1, n+1) * np.pi / (n+1))
    beta = np.sqrt((1+x) / (1-x))
    gamma = np.log(0.5 * (1-x))
    delta = (1+x)**alpha
    R0 = R / np.log(2)
    r = -R0 * delta * gamma
    w = np.pi / (n+1) * (R0 * delta)**3 * gamma**2 * (beta - alpha / beta * gamma)
    return r, w


def mura(n, R):
    x = np.arange(1, n+1) / (n+1)
    w = 3 * x**2 * np.log(1-x**3)**2 / (1-x**3) / (n+1) * R**3
    r = -R * np.log(1-x**3)
    return r, w


def de2(n, alpha, rmin, rmax):
    from scipy.optimize import root_scalar

    f = lambda x, r0: np.exp(alpha * x - np.exp(-x)) - r0
    xmin = root_scalar(lambda x: f(x, rmin), bracket=(-30, 30/alpha)).root
    xmax = root_scalar(lambda x: f(x, rmax), bracket=(-30, 30/alpha)).root

    x = np.linspace(xmin, xmax, n, endpoint=True)
    h = x[1] - x[0]
    ax = alpha * x
    emx = np.exp(-x)

    w = h * np.exp(3*(ax-emx)) * (alpha + emx)
    r = np.exp(ax - emx)
    return r, w


############################################################
#                       Test
############################################################
import unittest

from scipy.special import sph_harm

class TestRadial(unittest.TestCase):

    def setUp(self):
        a1 = 0.3
        a2 = 3.0

        self.funcs = [
                lambda r: np.exp(-a1 * r * r) + np.exp(-a2 * r * r),
                lambda r: r * (np.exp(-a1 * r * r) + np.exp(-a2 * r * r)),
                lambda r: r * r * (np.exp(-a1 * r * r) + np.exp(-a2 * r * r))
            ]

        self.refs = [
                0.25 * np.sqrt(np.pi) * (a1**(-1.5) + a2**(-1.5)),
                0.5 / a1**2 + 0.5 / a2**2,
                0.375 * np.sqrt(np.pi) * (a1**(-2.5) + a2**(-2.5))
            ]


    def test_gauss_laguerre(self):
        for func, ref in zip(self.funcs, self.refs):
            r, w = gauss_laguerre(40, 0.5)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)


    def test_becke(self):
        for func, ref in zip(self.funcs, self.refs):
            r, w = becke(40, 5.0)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)


    def test_murray(self):
        for func, ref in zip(self.funcs, self.refs):
            r, w = murray(40, 5.0)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)


    def test_baker(self):
        for func, ref in zip(self.funcs, self.refs):
            r, w = baker(20, 7.0, 2)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)


    def test_treutler_m4(self):
        for func, ref in zip(self.funcs, self.refs):
            r, w = treutler_m4(40, 5.0, 0.6)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)

            # m4 with alpha=0 is equivalent to m3
            r, w = treutler_m4(40, 5.0, 0.0)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)


    def test_mura(self):
        for func, ref in zip(self.funcs, self.refs):
            r, w = mura(40, 5.0)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)


    def test_de2(self):
        for func, ref in zip(self.funcs, self.refs):
            r, w = de2(40, 0.5, 1e-7, 10)
            val = np.sum(func(r) * w)
            self.assertAlmostEqual(val, ref, 6)


if __name__ == '__main__':
    unittest.main()


