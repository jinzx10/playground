import numpy as np
from harm import real_solid_harm
from addition import M_nz
from gaunt import real_gaunt

def gauss_prod(alpha, A, beta, B):
    '''
    Gaussian product rule.

    Returns the prefactor "K" and new center "C" of the product
    of two Gaussians:

        exp[-alpha*(r-A)^2] * exp[-beta*(r-B)^2]
            = K * exp[-(alpha+beta)*(r-C)^2]

    '''
    gamma = alpha * beta / (alpha + beta)
    rAB = np.linalg.norm(A-B)
    K = np.exp(-gamma*rAB**2)
    C = (alpha * A + beta * B) / (alpha + beta)
    return K, C


def real_solid_harm_prod(A, l1, m1, B, l2, m2, C):
    r'''
    Expansion of a product of two real solid harmonics.

    A product of an arbitrary pair of real solid harmonics
    can be expanded into a sum of real solid harmonics on
    an arbitrary new center:

     m1         m2        -- -- --             p    m
    S  (r-A) * S  (r-B) = \  \  \  coef * |r-C|  * S (r-C)
     l1         l2        /  /  /                   l
                          -- -- --
                          p  l  m

    This function returns the above expansion as a dict
    {(p, l, m): coef} 

    '''
    M1_nz = M_nz(l1, m1)
    M2_nz = M_nz(l2, m2)

    CA = C - A
    CB = C - B

    xpan = {}

    for (l1p, nu1, lam1), coef1 in M1_nz:
        S1 = real_solid_harm(l1-l1p, lam1, CA)
        for (l2p, nu2, lam2), coef2 in M2_nz:
            S2 = real_solid_harm(l2-l2p, lam2, CB)
            fac = S1 * S2 * 2 * np.sqrt(np.pi / ((2*l1p+1) * (2*l2p+1)))
            for l in range(abs(l1p-l2p), l1p+l2p+1, 2):
                for m in range(-l, l+1):
                    G = real_gaunt(l1p, l2p, l, nu1, nu2, m)
                    if G != 0.0:
                        key = (l1p+l2p-l, l, m)
                        val = coef1 * coef2 * fac * np.sqrt(2*l+1) * G
                        if key not in xpan:
                            xpan[key] = val
                        else:
                            xpan[key] += val

                        # (4, -2) x (4, 2)
                        #if key == (2, 4, 0):
                        #    print(f'l1p={l1p} nu1={nu1:2} lam1={lam1:2} '
                        #          f'l2p={l2p} nu2={nu2:2} lam2={lam2:2} '
                        #          f'val={val}')

    return xpan


def sGTO_prod(alpha, A, l1, m1, beta, B, l2, m2):
    '''
    Expansion of a product of two spherical GTOs.

    '''
    K, C = gauss_prod(alpha, A, beta, B)
    xpan = real_solid_harm_prod(A, l1, m1, B, l2, m2, C)
    return {key: coef * K for key, coef in xpan.items()}


#####################################################################

import unittest

class TestProd(unittest.TestCase):

    def test_real_solid_harm_prod(self):
        r = np.random.randn(3)
        A = np.random.randn(3)
        B = np.random.randn(3)
        C = np.random.randn(3)
        
        l1, m1 = 3, 1
        l2, m2 = 2, -2
        
        xpan = real_solid_harm_prod(A, l1, m1, B, l2, m2, C)
        rCabs = np.linalg.norm(r-C)
        val = sum(coef
                  * rCabs**key[0]
                  * real_solid_harm(key[1], key[2], r-C)
                  for key, coef in xpan.items())

        ref = real_solid_harm(l1, m1, r-A) * real_solid_harm(l2, m2, r-B)
        self.assertAlmostEqual(ref, val, 12)


    def test_sGTO_prod(self):

        def sgto(r, r0, a, l, m):
            rr = r - r0
            return np.exp(-a * np.linalg.norm(rr)**2) \
                    * real_solid_harm(l, m, rr)

        r = np.random.randn(3)
        A = np.random.randn(3)
        B = np.random.randn(3)

        alpha = np.random.rand()
        beta  = np.random.rand()

        gamma = alpha + beta
        C = gauss_prod(alpha, A, beta, B)[1]
        
        rC = r - C
        rCabs = np.linalg.norm(rC)

        l1, m1 = 4, 2
        l2, m2 = 4, 2

        #l1, l2 = np.random.randint(REAL_GAUNT_TABLE_LMAX+1, size=2)
        #m1 = np.random.randint(-l1, l1+1)
        #m2 = np.random.randint(-l2, l2+1)
        
        xpan = sGTO_prod(alpha, A, l1, m1, beta, B, l2, m2)
        val = sum(coef
                  * rCabs**key[0]
                  * sgto(r, C, gamma, key[1], key[2])
                  for key, coef in xpan.items())

        for key, coef in xpan.items():
            print(key, coef)

        ref = sgto(r, A, alpha, l1, m1) * sgto(r, B, beta, l2, m2)
        self.assertAlmostEqual(ref, val, 12)


if __name__ == '__main__':
    unittest.main()


