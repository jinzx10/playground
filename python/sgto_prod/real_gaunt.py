import numpy as np
import os

from scipy.io import savemat
from sympy import I, conjugate, Float
from sympy.physics.wigner import gaunt
#NOTE real_gaunt() in sympy is buggy, don't use it!

from harm import Y2R_sym, _ind, _rind


REAL_GAUNT_TABLE_LMAX = 4
REAL_GAUNT_TABLE = './real_gaunt.npy'


def real_gaunt_sym(l1, l2, l3, m1, m2, m3):
    '''
    Real Gaunt coefficients (symbolic).

    This function computes the real Gaunt coefficients by a brute-
    force transformation from the standard Gaunt coefficients.
    The sign convention of real spherical harmonics follows the
    convention in Helgaker's book.

    '''
    # selection rule
    if l1 + l2 < l3 or l1 + l3 < l2 or l2 + l3 < l1 or \
        (l1 + l2 + l3) % 2 or \
        ( (m1 < 0) + (m2 < 0) + (m3 < 0) ) % 2 or \
        ( abs(m1) + abs(m2) != abs(m3) and \
          abs(m2) + abs(m3) != abs(m1) and \
          abs(m3) + abs(m1) != abs(m2) ):
        return Float(0.0)

    val = 0
    for m1p in range(-l1, l1+1):
        u1 = Y2R_sym(m1, m1p)
        for m2p in range(-l2, l2+1):
            u2 = Y2R_sym(m2, m2p)
            for m3p in range(-l3, l3+1):
                u3 = Y2R_sym(m3, m3p)
                val += u1 * u2 * u3 * gaunt(l1, l2, l3, m1p, m2p, m3p)

    return val


def real_gaunt_gen(fname, lmax1, lmax2, lmax3):
    '''
    Tabulate and store the real Gaunt coefficients.

    Index map: (l,m) -> l*l + l + m
    Reverse index map: i -> (l=int(sqrt(i)), m=i-l*l-l)

    '''
    table = np.zeros(((lmax1+1)**2, (lmax2+1)**2, (lmax3+1)**2))

    print(f'Generate real Gaunt table up to '
          f'l1={lmax1}, l2={lmax2}, l3={lmax3} ...')

    for i1 in range((lmax1+1)**2):
        l1, m1 = _rind(i1)
        print(f'{i1+1}/{(lmax1+1)**2}')
        for i2 in range((lmax2+1)**2):
            l2, m2 = _rind(i2)
            for i3 in range((lmax3+1)**2):
                l3, m3 = _rind(i3)
                table[i1, i2, i3] = \
                        float(real_gaunt_sym(l1, l2, l3, m1, m2, m3))

    np.save(fname, table)
    savemat(fname.replace('.npy', '.mat'),
            {'real_gaunt_table': table}, appendmat=False)


if not os.path.isfile(REAL_GAUNT_TABLE):
    real_gaunt_gen(REAL_GAUNT_TABLE,
                   REAL_GAUNT_TABLE_LMAX,
                   REAL_GAUNT_TABLE_LMAX,
                   REAL_GAUNT_TABLE_LMAX*2)

_real_gaunt_table = np.load(REAL_GAUNT_TABLE)

def real_gaunt(l1, l2, l3, m1, m2, m3):
    return _real_gaunt_table[_ind(l1,m1), _ind(l2,m2), _ind(l3,m3)]


##################################################################


import unittest

from harm import real_sph_harm

class TestRealGaunt(unittest.TestCase):

    def test_sym(self):
        r'''
        Checks real_gaunt_sym by verifying that they do serve
        as the expansion coefficients for a product of two real
        spherical harmonics:

         m1  m2   -- --   m1 m2 m3    m3
        Y   Y   = \  \   G         * Y
         l1  l2   /  /    l1 l2 l3    l3
                  -- --
                  l3 m3

        '''

        lmax = 2

        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        for l1 in range(lmax+1):
            for m1 in range(-l1, l1+1):
                Y1 = real_sph_harm(l1, m1, theta, phi)
                for l2 in range(lmax+1):
                    for m2 in range(-l2, l2+1):
                        Y2 = real_sph_harm(l2, m2, theta, phi)

                        ref = Y1 * Y2
                        val = sum(real_gaunt_sym(l1, l2, l3, m1, m2, m3)
                                  * real_sph_harm(l3, m3, theta, phi)
                                  for l3 in range(0, l1+l2+1)
                                  for m3 in range(-l3, l3+1))

                        self.assertAlmostEqual(float(val), ref, 12)


    
    def test_table(self):
        '''
        Checks _real_gaunt_table by the same test as above.

        '''
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        for l1 in range(REAL_GAUNT_TABLE_LMAX+1):
            for m1 in range(-l1, l1+1):
                Y1 = real_sph_harm(l1, m1, theta, phi)
                for l2 in range(REAL_GAUNT_TABLE_LMAX+1):
                    for m2 in range(-l2, l2+1):
                        Y2 = real_sph_harm(l2, m2, theta, phi)

                        val = sum(_real_gaunt_table[_ind(l1,m1),
                                                    _ind(l2,m2),
                                                    _ind(l3,m3)]
                                  * real_sph_harm(l3, m3, theta, phi)
                                  for l3 in range(0, l1+l2+1)
                                  for m3 in range(-l3, l3+1))

                        ref = Y1 * Y2
                        self.assertAlmostEqual(float(val), ref, 12)


if __name__ == '__main__':
    unittest.main()

