import os
import numpy as np

from math import comb
from sympy import sqrt
from scipy.io import savemat
from scipy.sparse import csr_matrix, save_npz, load_npz

from harm import real_solid_harm, R2Y_sym, Y2R_sym, pack_lm, unpack_lm


REAL_ADDITION_TABLE_LMAX = 4
REAL_ADDITION_TABLE = './real_addition_table.npz'

np.set_printoptions(legacy='1.25')


def M_sym(l, mu, lp, nu, lam):
    r'''
    Symbolic coefficients in the real solid harmonics' addition theorem.
    (abbreviated as "M coefficients" hereafter)

                 l     lp      l-lp
     mu          --    --       --     mu nu lam    nu        lam
    S  (r1+r2) = \     \        \     M          * S  (r1) * S   (r2)
     l           /     /        /      l  lp        lp        l-lp
                 --    --       --
                lp=0  nu=-lp  lam=lp-l

    '''
    return sum(Y2R_sym(mu, m) * R2Y_sym(mp, nu) * R2Y_sym(m-mp, lam)
               * sqrt(comb(l+m, lp+mp) * comb(l-m, lp-mp))
               for m in range(-l, l+1)
               for mp in range(max(-lp, m+lp-l), min(lp,m+l-lp)+1)
               )


def pack_M(lp, nu, lam, lmax=REAL_ADDITION_TABLE_LMAX):
    return pack_lm(lp, nu) * (2*lmax+1) + lam + lmax


def unpack_M(col, lmax=REAL_ADDITION_TABLE_LMAX):
    r, _lam = divmod(col, 2*lmax+1)
    lp, nu = unpack_lm(r)
    return lp, nu, _lam - lmax


def M_gen(fname, lmax):
    '''
    Tabulate and save the M coefficients to file.

    M is made a 2-d sparse matrix by grouping its indices as

            (l,mu) x (lp,nu,lam)

    See pack_lm()/unpack_lm() for the index map of (l,mu).
    See pack_M()/unpack_M() for the index map of (lp,nu,lam).

    '''
    table = np.zeros(((lmax+1)**2, (lmax+1)**2*(2*lmax+1)))

    print(f'Generate real addition table up to l={lmax} ...')

    for i1 in range((lmax+1)**2):
        l, mu = unpack_lm(i1)
        print(f'{i1+1}/{(lmax+1)**2}', end='\r')
        for r in range((lmax+1)**2):
            lp, nu = unpack_lm(r)
            if lp > l:
                break
            for lam in range(lp-l, l-lp+1):
                i2 = pack_M(lp, nu, lam, lmax)
                table[i1, i2] = float(M_sym(l, mu, lp, nu, lam))
    print('')

    table_csr = csr_matrix(table)

    save_npz(fname, table_csr)

    # MATLAB uses CSC format, so it's better to transpose
    savemat(fname.replace('.npz', '.mat'),
            {'REAL_ADDITION_TABLE': table_csr.transpose(),
             'REAL_ADDITION_TABLE_LMAX': float(lmax)})


if not os.path.isfile(REAL_ADDITION_TABLE):
    M_gen(REAL_ADDITION_TABLE, REAL_ADDITION_TABLE_LMAX)

_real_addition_table = load_npz(REAL_ADDITION_TABLE)

def M(l, mu, lp, nu, lam):
    return _real_addition_table[pack_lm(l,mu), pack_M(lp, nu, lam)]


def M_nz(l, mu):
    '''
    Returns all non-zero M coefficients with the given (l, mu) as
    a list of pairs ((lp,nu,lam), coef).

    '''
    i1 = pack_lm(l,mu)
    row = _real_addition_table[i1]
    lp_nu_lam = [unpack_M(i2) for i2 in row.indices]
    return list(zip(lp_nu_lam, row.data))


########################################################################

import unittest

class TestAddition(unittest.TestCase):

    def test_M_sym(self):
        '''
        Checks M_sym by verifying the addition theorem.

        '''
        lmax = 3
        for l in range(lmax+1):
            for m in range(-l, l+1):
                r1 = np.random.randn(3)
                r2 = np.random.randn(3)

                val = sum(M_sym(l, m, lp, nu, lam)
                          * real_solid_harm(lp, nu, r1)
                          * real_solid_harm(l-lp, lam, r2)
                          for lp in range(l+1)
                          for nu in range(-lp, lp+1)
                          for lam in range(lp-l, l-lp+1))

                ref = real_solid_harm(l, m, r1+r2)
                self.assertAlmostEqual(ref, float(val), 12)


    def test_M(self):
        '''
        Checks the tabulated version with the same test as above.

        '''
        for l in range(REAL_ADDITION_TABLE_LMAX+1):
            for m in range(-l, l+1):
                r1 = np.random.randn(3)
                r2 = np.random.randn(3)

                val = sum(M(l, m, lp, nu, lam)
                          * real_solid_harm(lp, nu, r1)
                          * real_solid_harm(l-lp, lam, r2)
                          for lp in range(l+1)
                          for nu in range(-lp, lp+1)
                          for lam in range(lp-l, l-lp+1))

                ref = real_solid_harm(l, m, r1+r2)
                self.assertAlmostEqual(ref, val, 12)


    def test_M_nz(self):
        '''
        Checks the tabulated version with the same test as above.

        '''
        for l in range(REAL_ADDITION_TABLE_LMAX+1):
            for m in range(-l, l+1):
                r1 = np.random.randn(3)
                r2 = np.random.randn(3)

                M_list = M_nz(l, m)
                val = sum(coef
                          * real_solid_harm(lp, nu, r1)
                          * real_solid_harm(l-lp, lam, r2)
                          for ((lp, nu, lam), coef) in M_list)

                ref = real_solid_harm(l, m, r1+r2)
                self.assertAlmostEqual(ref, val, 12)


if __name__ == '__main__':
    unittest.main()


