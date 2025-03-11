import os
import numpy as np

from math import comb
from sympy import sqrt
from scipy.io import savemat

from harm import real_solid_harm, R2Y_sym, Y2R_sym, _ind, _rind


REAL_ADDITION_TABLE_LMAX = 4
REAL_ADDITION_TABLE = './real_addition.npy'


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

    '''
    table = np.zeros(((lmax+1)**2, (lmax+1)**2, 2*lmax+1))

    print(f'Generate M table up to l={lmax} ...')

    for i1 in range((lmax+1)**2):
        l, mu = _rind(i1)
        print(f'{i1+1}/{(lmax+1)**2}')
        for i2 in range((lmax+1)**2):
            lp, nu = _rind(i2)
            if lp > l:
                break
            for lam in range(lp-l, l-lp+1):
                i3 = lam + REAL_ADDITION_TABLE_LMAX
                table[i1, i2, i3] = float(M_sym(l, mu, lp, nu, lam))

    np.save(fname, table)
    savemat(fname.replace('.npy', '.mat'),
            {'real_addition_table': table}, appendmat=False)


if not os.path.isfile(REAL_ADDITION_TABLE):
    M_gen(REAL_ADDITION_TABLE, REAL_ADDITION_TABLE_LMAX)


_real_addition_table = np.load(REAL_ADDITION_TABLE)

def M(l, mu, lp, nu, lam):
    return _real_addition_table[_ind(l,mu), _ind(lp,nu),
                                lam + REAL_ADDITION_TABLE_LMAX]


def M_all(l, mu=None):
    '''
    Non-zero M coefficients.

    '''
    if mu is not None:

    coef_all = []
    for lp in range(l+1):
        for nu in range(-lp, lp+1):
            for lam in range(lp-l, l-lp+1):
                coef = M(l, mu, lp, nu, lam)
                if coef != 0:
                    coef_all.append(((lp, nu, lam), float(coef)))

    return coef_all


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


if __name__ == '__main__':
    unittest.main()


