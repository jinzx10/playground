import os
import numpy as np

from math import comb
from sympy import sqrt
from scipy.io import savemat
from scipy.sparse import csr_matrix, save_npz, load_npz

from harm import real_solid_harm, R2Y_sym, Y2R_sym, _ind, _rind


REAL_ADDITION_TABLE_LMAX = 4
REAL_ADDITION_TABLE = './real_addition.npz'

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


def M_gen(fname, lmax):
    '''
    Tabulate and save the M coefficients to file.

    M is made a 2-d array by grouping its indices as

            (l,m) x (lp,nu,lam)

    '''
    table = np.zeros(((lmax+1)**2, (lmax+1)**2*(2*lmax+1)))

    print(f'Generate M table up to l={lmax} ...')

    for i1 in range((lmax+1)**2):
        l, mu = _rind(i1)
        print(f'{i1+1}/{(lmax+1)**2}')
        for r in range((lmax+1)**2):
            lp, nu = _rind(r)
            if lp > l:
                break
            for lam in range(lp-l, l-lp+1):
                c = lam + REAL_ADDITION_TABLE_LMAX
                i2 = r * (2*lmax+1) + c
                table[i1, i2] = float(M_sym(l, mu, lp, nu, lam))

    table_csr = csr_matrix(table)

    save_npz(fname, table_csr)
    #savemat(fname.replace('.npz', '.mat'),
    #        {'real_addition_table': table}, appendmat=False)


if not os.path.isfile(REAL_ADDITION_TABLE):
    M_gen(REAL_ADDITION_TABLE, REAL_ADDITION_TABLE_LMAX)


_real_addition_table = load_npz(REAL_ADDITION_TABLE)

def M(l, mu, lp, nu, lam):
    i1 = _ind(l,mu)
    i2 = _ind(lp, nu) * (2*REAL_ADDITION_TABLE_LMAX+1) \
            + lam + REAL_ADDITION_TABLE_LMAX
    return _real_addition_table[i1, i2]


def colind_decode(col):
    r = col // (2*REAL_ADDITION_TABLE_LMAX+1)
    lp, nu = _rind(r)
    lam = col % (2*REAL_ADDITION_TABLE_LMAX+1) - REAL_ADDITION_TABLE_LMAX
    return lp, nu, lam


def M_all(l, mu=None):
    '''
    Non-zero M coefficients.

    '''
    if mu is not None:
        i1 = _ind(l,mu)
        row = _real_addition_table[i1]
        lp_nu_lam = [colind_decode(colind) for colind in row.indices]
        return list(zip(lp_nu_lam, row.data))
    else:
        irange = range(_ind(l,-l), _ind(l,l)+1)
        rows = _real_addition_table[irange]
        lp_nu_lam = [colind_decode(colind) for colind in rows.indices]
        print(lp_nu_lam)
        print(len(lp_nu_lam))
        print(len(set(lp_nu_lam)))
        pass



########################################################################

import unittest

class TestAddition(unittest.TestCase):

    def test_one(self):
        l = 4
        m = -3

        r1 = np.random.randn(3)
        r2 = np.random.randn(3)
        ref = real_solid_harm(l, m, r1+r2)
        
        val = 0.0
        for lp in range(l+1):
            for nu in range(-lp, lp+1):
                for lam in range(lp-l, l-lp+1):
                    MM = M(l, m, lp, nu, lam)
                    #print(f'l={l}  m={m:2}  lp={lp}  nu={nu:2}  lam={lam:2}  M^2={MM**2:10.5f}')
                    val += M(l, m, lp, nu, lam) * real_solid_harm(lp, nu, r1) * real_solid_harm(l-lp, lam, r2)
        
        #print(f'ref={ref: 20.15f}  val={val: 20.15f}  diff={abs(ref-val): 20.15f}')
        self.assertAlmostEqual(ref, val, 12)


    def test_many_lm(self):
        lmax = 4
        for l in range(lmax+1):
            for m in range(-l, l+1):
                r1 = np.random.randn(3)
                r2 = np.random.randn(3)
                ref = real_solid_harm(l, m, r1+r2)
                val = sum(M(l, m, lp, nu, lam) * real_solid_harm(lp, nu, r1) * real_solid_harm(l-lp, lam, r2)
                          for lp in range(l+1)
                          for nu in range(-lp, lp+1)
                          for lam in range(lp-l, l-lp+1))
                self.assertAlmostEqual(ref, val, 12)


    def test_all(self):
        l = 4
        m = -3

        r1 = np.random.randn(3)
        r2 = np.random.randn(3)
        ref = real_solid_harm(l, m, r1+r2)

        M_list = M_all(l, m)
        val = sum(coef * real_solid_harm(lp, nu, r1) * real_solid_harm(l-lp, lam, r2)
                  for ((lp, nu, lam), coef) in M_list)
        
        self.assertAlmostEqual(ref, val, 12)


    def test_all2(self):
        l = 4
        m = -3

        r1 = np.random.randn(3)
        r2 = np.random.randn(3)
        ref = real_solid_harm(l, m, r1+r2)

        M_list = M_all(l)
        #val = sum(coef * real_solid_harm(lp, nu, r1) * real_solid_harm(l-lp, lam, r2)
        #          for ((lp, nu, lam), coef) in M_list)
        

if __name__ == '__main__':
    unittest.main()


