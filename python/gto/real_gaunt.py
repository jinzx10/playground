import numpy as np
import sympy as sp

from sympy.physics.wigner import gaunt
from sympy import I

#NOTE real_gaunt() in sympy is buggy, don't use it!

def r2s(m, mp):
    # spherical harmonics: real-to-standard transformation
    return int((m == 0) * (mp == 0)) + 1/sp.sqrt(2) * (
            int(m > 0) * (-1)**m * (int(m == mp) + I * int(m == -mp)) + 
            int(m < 0) * (int(m == -mp) - I * int(m == mp))
            )


def s2r(m, mp):
    # standard-to-real
    return np.conj(r2s(mp, m))


def real_gaunt(l1, l2, l3, m1, m2, m3):
    r'''
    Real Gaunt coefficients

    This function computes the real Gaunt coefficients by a brute-
    force transformation from the standard Gaunt coefficients.
    The sign convention of real spherical harmonics follows the
    convention in Helgaker's book.

    '''
    val = 0
    for m1p in range(-l1, l1+1):
        u1 = s2r(m1, m1p)
        for m2p in range(-l2, l2+1):
            u2 = s2r(m2, m2p)
            for m3p in range(-l3, l3+1):
                u3 = s2r(m3, m3p)
                val += u1 * u2 * u3 * gaunt(l1, l2, l3, m1p, m2p, m3p)

    #return val
    return float(val.evalf())


#lmax = 2
#for l1 in range(lmax+1):
#    for l2 in range(lmax+1):
#        for l3 in range(2*lmax+1):
#            for m1 in range(-l1, l1+1):
#                for m2 in range(-l2, l2+1):
#                    for m3 in range(-l3, l3+1):
#                        tmp = real_gaunt(l1, l2, l3, m1, m2, m3)
#                        if tmp != 0:
#                            print(f'l1={l1}  l2={l2}  l3={l3}  m1={m1:2}  m2={m2:2}  m3={m3:2}  G={tmp:20.15f}')




#######################################################################

import unittest

from harm import real_sph_harm

class TestRealGaunt(unittest.TestCase):

    def _test_expand(self):
        r'''
        Verifies that real Gaunt coefficients do serve as the expansion
        coefficients for a product of two real spherical harmonics:

         m1  m2   -- --   m1 m2 m3    m3
        Y   Y   = \  \   G         * Y
         l1  l2   /  /    l1 l2 l3    l3
                  -- --
                  l3 m3

        '''

        lmax = 3
        n = 1

        for i in range(n):
            theta = np.random.rand() * np.pi
            phi = np.random.rand() * 2 * np.pi

            for l1 in range(lmax+1):
                for m1 in range(-l1, l1+1):
                    Y1 = real_sph_harm(l1, m1, theta, phi)
                    for l2 in range(lmax+1):
                        for m2 in range(-l2, l2+1):
                            Y2 = real_sph_harm(l2, m2, theta, phi)

                            val = 0
                            for l3 in range(abs(l1-l2), l1+l2+1, 2):
                                for m3 in range(-l3, l3+1):
                                    Y3 = real_sph_harm(l3, m3, theta, phi)
                                    val += real_gaunt(l1, l2, l3, m1, m2, m3) * Y3
                            ref = Y1 * Y2
                            self.assertAlmostEqual(val, ref, 12)




if __name__ == '__main__':
    print(real_gaunt(1,1,0,1,1,0))
    unittest.main()

