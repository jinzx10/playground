import numpy as np
from math import comb
from sympy import sqrt
from harm import real_sph_harm, real_solid_harm, \
        R2Y, Y2R, R2Y_sym, Y2R_sym


def M(l, mu, lp, nu, lam):
    r'''
    Coefficients in the real solid harmonics' addition theorem.

                 l     lp      l-lp
     mu          --    --       --                         nu        lam
    S  (r1+r2) = \     \        \     M(l,lp,mu,nu,lam) * S  (r1) * S   (r2)
     l           /     /        /                          lp        l-lp
                 --    --       --
                lp=0  nu=-lp  lam=lp-l

    '''
    return np.real(sum(
        Y2R(mu, m) * R2Y(mp, nu) * R2Y(m-mp, lam)
        * np.sqrt(comb(l+m, lp+mp) * comb(l-m, lp-mp))
        for m in range(-l, l+1)
        for mp in range(max(-lp, m+lp-l), min(lp,m+l-lp)+1)
        ))


def M_all(l, mu):
    '''
    Non-zero coefficients in the real solid harmonics' addition theorem.
    See also M().

    '''
    coef_all = []
    for lp in range(l+1):
        for nu in range(-lp, lp+1):
            for lam in range(lp-l, l-lp+1):
                coef = M(l, mu, lp, nu, lam)
                if coef != 0:
                    coef_all.append(((lp, nu, lam), float(coef)))

    return coef_all


def M_sym(l, mu, lp, nu, lam):
    '''
    Symbolic version of M().

    '''
    return sum(Y2R_sym(mu, m) * R2Y_sym(mp, nu) * R2Y_sym(m-mp, lam)
               * sqrt(comb(l+m, lp+mp) * comb(l-m, lp-mp))
               for m in range(-l, l+1)
               for mp in range(max(-lp, m+lp-l), min(lp,m+l-lp)+1)
               )


def M_all_sym(l, mu):
    '''
    Symbolic version of M_all().

    '''
    coef_all = []
    for lp in range(l+1):
        for nu in range(-lp, lp+1):
            for lam in range(lp-l, l-lp+1):
                coef = M_sym(l, mu, lp, nu, lam)
                if coef != 0:
                    coef_all.append(((lp, nu, lam), coef))

    return coef_all

###########################################################################

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
    #print(M_sym(4, 3, 2, 2, 1))
    #print(M_all_sym(4, 3))
    unittest.main()


