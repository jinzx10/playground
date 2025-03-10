import numpy as np
import sympy as sp

from scipy.special import sph_harm_y
from sympy.physics.wigner import gaunt
from sympy import I, conjugate, Float

#NOTE real_gaunt() in sympy is buggy, don't use it!

def real_sph_harm(l, m, theta, phi):
    '''
    Real spherical harmonics.

    The convention follows that of the Helgaker's book.

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

    The convention follows that of the Helgaker's book.

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


'''
Index map:
forward : (l,m) -> l*l + l + m
backward: i -> (l=int(sqrt(i)), m=i-l*l-l)

'''
def _ind(l, m):
    return l*l + l + m


def _rind(i):
    l = int(np.sqrt(i))
    m = i - l*l - l
    return l, m


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


if __name__ == '__main__':
    unittest.main()




