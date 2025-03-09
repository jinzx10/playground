import numpy as np
import sympy as sp

from scipy.special import sph_harm_y
from sympy.physics.wigner import gaunt
from sympy import I, conjugate, Float

#NOTE real_gaunt() in sympy is buggy, don't use it!

def real_sph_harm(l, m, theta, phi):
    '''
    Real spherical harmonics.

    The sign convention follows that of the Helgaker's book.

    '''
    if m == 0:
        return np.real(sph_harm_y(l, 0, theta, phi))
    elif m > 0:
        return (-1)**m * np.sqrt(2) * np.real(sph_harm_y(l, m, theta, phi))
    else:
        return (-1)**m * np.sqrt(2) * np.imag(sph_harm_y(l, -m, theta, phi))


def real_solid_harm(l, m, r):
    '''
    Real solid harmonics.

    The sign & normalization convention follows that of the Helgaker's book.

    '''
    rabs = np.linalg.norm(r)
    theta = np.arccos(r[2]/rabs)
    phi = np.arctan2(r[1], r[0])
    return np.sqrt(4*np.pi/(2*l+1)) * rabs**l \
            * real_sph_harm(l, m, theta, phi)


def R2Y(m, mp):
    r'''
    Real-to-standard transformation of spherical harmonics.

    Let Y be the standard spherical harmonics and R be the real
    spherical harmonics following the convention of Helgaker's book,
    this function returns the transformation matrix element such that

             l 
         m   --                mp
        Y  = \    R2Y(m,mp) * R
         l   /                 l    
             --
            mp=-l
    
    '''
    return (m == 0) * (mp == 0) + 1/np.sqrt(2) * (
            (m > 0) * (-1)**m * ((m == mp) + 1j * (m == -mp)) + 
            (m < 0) * ((m == -mp) - 1j * (m == mp))
            )


def Y2R(m, mp):
    r'''
    Standard-to-real transformation of spherial harmonics.

             l 
         m   --                mp
        R  = \    Y2R(m,mp) * Y
         l   /                 l    
             --
            mp=-l

    See also R2Y().

    '''
    return np.conj(R2Y(mp, m))


def R2Y_sym(m, mp):
    '''
    Symbolic version of R2Y.

    '''
    return int((m == 0) * (mp == 0)) + 1/sp.sqrt(2) * (
            int(m > 0) * (-1)**m * (int(m == mp) + I * int(m == -mp)) + 
            int(m < 0) * (int(m == -mp) - I * int(m == mp))
            )


def Y2R_sym(m, mp):
    '''
    Symbolic version of Y2R.

    '''
    return conjugate(R2Y_sym(mp, m))


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


def real_gaunt(l1, l2, l3, m1, m2, m3):
    return float(real_gaunt_sym(l1, l2, l3, m1, m2, m3).evalf())


##############################################################

import unittest

class TestHarm(unittest.TestCase):

    def test_R2Y(self):
        lmax = 8
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * np.pi * 2

        for l in range(lmax+1):
            for m in range(-l, l+1):
                ref = sph_harm_y(l, m, theta, phi)
                val = sum(R2Y(m, mp) * real_sph_harm(l, mp, theta, phi)
                          for mp in range(-l, l+1))
                self.assertAlmostEqual(ref, val, 12)


    def test_Y2R(self):
        lmax = 8
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * np.pi * 2

        for l in range(lmax+1):
            for m in range(-l, l+1):
                ref = real_sph_harm(l, m, theta, phi)
                val = sum(Y2R(m, mp) * sph_harm_y(l, mp, theta, phi)
                          for mp in range(-l, l+1))
                self.assertAlmostEqual(ref, val, 12)

    def test_expand(self):
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

        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        for l1 in range(lmax+1):
            for m1 in range(-l1, l1+1):
                Y1 = real_sph_harm(l1, m1, theta, phi)
                for l2 in range(lmax+1):
                    for m2 in range(-l2, l2+1):
                        Y2 = real_sph_harm(l2, m2, theta, phi)

                        ref = Y1 * Y2
                        val = sum(real_gaunt(l1, l2, l3, m1, m2, m3)
                                  * real_sph_harm(l3, m3, theta, phi)
                                  #for l3 in range(abs(l1-l2), l1+l2+1, 2)
                                  for l3 in range(0, l1+l2+1)
                                  for m3 in range(-l3, l3+1))

                        self.assertAlmostEqual(val, ref, 12)


    def _test_print_table(self):
        lmax = 3
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for l3 in range(2*lmax+1):
                    for m1 in range(-l1, l1+1):
                        for m2 in range(-l2, l2+1):
                            for m3 in range(-l3, l3+1):
                                G = real_gaunt_sym(l1,l2,l3,m1,m2,m3)
                                if G != 0:
                                    print(f'l1={l1}  l2={l2}  l3={l3}  m1={m1:2}  m2={m2:2}  m3={m3:2}  ', G)


if __name__ == '__main__':
    unittest.main()




