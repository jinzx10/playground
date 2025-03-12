import numpy as np
import os

from scipy.io import savemat
from scipy.sparse import csr_matrix, save_npz, load_npz
from sympy import Float
from sympy.physics.wigner import gaunt
#NOTE real_gaunt() in sympy is buggy, don't use it!

from harm import Y2R_sym, pack_lm, unpack_lm


REAL_GAUNT_TABLE_LMAX = 4
REAL_GAUNT_TABLE = './real_gaunt_table.npz'


def gaunt_select_l(l1, l2, l3):
    '''
    Selection rule of l for the (real) Gaunt coefficients.

    '''
    return l1 + l2 >= l3 and l2 + l3 >= l1 and l3 + l1 >= l1 \
            and (l1 + l2 + l3) % 2 == 0


def real_gaunt_select_m(m1, m2, m3):
    '''
    Selection rule of m for the REAL Gaunt coefficients.

    '''
    return  ( (m1 < 0) + (m2 < 0) + (m3 < 0) ) % 2 == 0 and \
            ( abs(m1) + abs(m2) == abs(m3) or \
              abs(m2) + abs(m3) == abs(m1) or \
              abs(m3) + abs(m1) == abs(m2) );


def real_gaunt_sym(l1, l2, l3, m1, m2, m3):
    r'''
    Real Gaunt coefficients (symbolic).

    This function computes the real Gaunt coefficients by a brute-
    force transformation from the standard Gaunt coefficients.
    The sign convention of real spherical harmonics follows the
    convention in Helgaker's book. The resulting coefficients
    serve as the expansion coefficients for a product of two real
    spherical harmonics:

                    l1+l2      l3
         m1  m2      --        --   m1 m2 m3    m3
        R   R   =    \         \   G         * R
         l1  l2      /         /    l1 l2 l3    l3
                     --        --
                 l3=|l1-l2|  m3=-l3

    '''
    assert(abs(m1) <= l1 and abs(m2) <= l2 and abs(m3) <= l3)

    if not (gaunt_select_l(l1, l2, l3) and
            real_gaunt_select_m(m1, m2, m3)):
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


def pack_G(l1, m1, l2, m2, lmax=REAL_GAUNT_TABLE_LMAX):
    return pack_lm(l1, m1) * (lmax+1)**2 + pack_lm(l2, m2)


def unpack_G(row, lmax=REAL_GAUNT_TABLE_LMAX):
    i1, i2 = divmod(row, (lmax+1)**2)
    return *unpack_lm(i1), *unpack_lm(i2)


def real_gaunt_gen(fname, lmax=REAL_GAUNT_TABLE_LMAX):
    '''
    Tabulate and save the real Gaunt coefficients to file.

    This function generates a table of real Gaunt coefficients
    adequate for the expansion of a product of two real spherical
    harmonics each of which has an angular momentum of lmax or less.
    That is to say, the maximal angular momentum of the table is
    (lmax, lmax, 2*lmax).

    This table is made a 2-d sparse matrix by grouping its indices as

            (l1,m1,l2,m2) x (l3,m3)

    See pack_lm()/unpack_lm() for the index map of (l3,m3).
    See pack_G()/unpack_G() for the index map of (l1,m1,l2,m2).

    '''
    table = np.zeros(((lmax+1)**4, (2*lmax+1)**2))

    print(f'Generate real Gaunt table up to l={lmax}...')

    for row in range((lmax+1)**4):
        l1, m1, l2, m2 = unpack_G(row)
        print(f'{row+1}/{(lmax+1)**4}', end='\r')
        for l3 in range(abs(l1-l2), l1+l2+1, 2):
            for m3 in [x for x in {m1+m2, m1-m2, m2-m1, -m1-m2}
                       if abs(x) <= l3]:
                col = pack_lm(l3, m3)
                table[row, col] = \
                        float(real_gaunt_sym(l1, l2, l3, m1, m2, m3))
    print('')
    table_csr = csr_matrix(table)
    save_npz(fname, table_csr)

    # MATLAB uses CSC format, so it's better to transpose
    savemat(fname.replace('.npz', '.mat'),
            {'REAL_GAUNT_TABLE': table_csr.transpose(),
             'REAL_GAUNT_TABLE_LMAX': float(lmax)})


if not os.path.isfile(REAL_GAUNT_TABLE):
    real_gaunt_gen(REAL_GAUNT_TABLE, REAL_GAUNT_TABLE_LMAX)

_real_gaunt_table = load_npz(REAL_GAUNT_TABLE)

def real_gaunt(l1, l2, l3, m1, m2, m3):
    return _real_gaunt_table[pack_G(l1,m1,l2,m2), pack_lm(l3,m3)]


def real_gaunt_nz(l1, l2, m1, m2):
    '''
    Returns all non-zero real Gaunt coefficients with the given
    (l1,l2,m1,m2) as a list of pairs ((l3,m3), coef).

    '''
    ir = pack_G(l1,m1,l2,m2)
    tab = _real_gaunt_table[ir]
    l3m3 = [unpack_lm(ic) for ic in tab.indices]
    return list(zip(l3m3, tab.data))


##################################################################


import unittest

from harm import real_sph_harm

class TestRealGaunt(unittest.TestCase):

    def test_sym(self):
        '''
        Checks real_gaunt_sym by verifying that they do serve
        as the expansion coefficients for a product of two real
        spherical harmonics.

        '''

        lmax = 2

        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        for l1 in range(lmax+1):
            for m1 in range(-l1, l1+1):
                R1 = real_sph_harm(l1, m1, theta, phi)
                for l2 in range(lmax+1):
                    for m2 in range(-l2, l2+1):
                        R2 = real_sph_harm(l2, m2, theta, phi)

                        val = sum(real_gaunt_sym(l1, l2, l3, m1, m2, m3)
                                  * real_sph_harm(l3, m3, theta, phi)
                                  for l3 in range(0, l1+l2+1)
                                  for m3 in range(-l3, l3+1))

                        ref = R1 * R2
                        self.assertAlmostEqual(float(val), ref, 12)


    def test_table(self):
        '''
        Checks the tabulated version with the same test as above.

        '''
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        for l1 in range(REAL_GAUNT_TABLE_LMAX+1):
            for m1 in range(-l1, l1+1):
                R1 = real_sph_harm(l1, m1, theta, phi)
                for l2 in range(REAL_GAUNT_TABLE_LMAX+1):
                    for m2 in range(-l2, l2+1):
                        R2 = real_sph_harm(l2, m2, theta, phi)

                        val = sum(real_gaunt(l1, l2, l3, m1, m2, m3)
                                  * real_sph_harm(l3, m3, theta, phi)
                                  for l3 in range(0, l1+l2+1)
                                  for m3 in range(-l3, l3+1))

                        ref = R1 * R2
                        self.assertAlmostEqual(float(val), ref, 12)


    def test_table_nz(self):
        '''
        Checks the tabulated version with the same test as above.

        '''
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        for l1 in range(REAL_GAUNT_TABLE_LMAX+1):
            for m1 in range(-l1, l1+1):
                R1 = real_sph_harm(l1, m1, theta, phi)
                for l2 in range(REAL_GAUNT_TABLE_LMAX+1):
                    for m2 in range(-l2, l2+1):
                        R2 = real_sph_harm(l2, m2, theta, phi)

                        G_list = real_gaunt_nz(l1, l2, m1, m2)
                        val = sum(coef
                                  * real_sph_harm(l3, m3, theta, phi)
                                  for ((l3, m3), coef) in G_list)

                        ref = R1 * R2
                        self.assertAlmostEqual(float(val), ref, 12)

if __name__ == '__main__':
    unittest.main()

