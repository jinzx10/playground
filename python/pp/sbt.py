import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

def sbt(l, frk, r, q, k=0):
    '''
    Given frk = f(r) * r^k, compute the l-th order spherical Bessel
    transform of f(r) at q:

               / +inf
        g(q) = |       dr r^(2-k) j_l(q*r) [f(r)*r^k]
               / 0

    by Simpson integration.

    '''
    return simpson(r**(2-k) * frk * spherical_jn(l, q*r), r)


##############################################################

import unittest
import matplotlib.pyplot as plt

class _Test(unittest.TestCase):

    def test_sbt(self):
        r = np.linspace(0, 10, 100)
        f = r**2 * np.exp(-r*r)
        
        q = np.linspace(0, 5, 100)
        l = 2
        
        g_ref = np.sqrt(2)/16 * q*q * np.exp(-q*q/4)
        g = np.array([sbt(l, f, r, qi) for qi in q]) \
                * np.sqrt(2/np.pi)

        max_diff = np.max(np.abs(g - g_ref))
        self.assertLess(max_diff, 1e-8)
        
        #plt.plot(q, g)
        #plt.plot(q, g_ref, '--')
        #plt.show()


if __name__ == '__main__':
    unittest.main()

