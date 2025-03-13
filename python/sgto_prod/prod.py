import os
import numpy as np

from scipy.sparse import csr_matrix, save_npz, load_npz

from harm import real_solid_harm, pack_lm, unpack_lm
from addition import M_nz, REAL_ADDITION_TABLE_LMAX
from gaunt import real_gaunt_nz, REAL_GAUNT_TABLE_LMAX

MMG_TABLE_LMAX = 4
MMG_TABLE = './MMG_table.npz'

np.set_printoptions(legacy='1.25')

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


def pack_lmlm(l1, m1, l2, m2, lmax=MMG_TABLE_LMAX):
    return pack_lm(l1, m1) * (lmax+1)**2 + pack_lm(l2, m2)


def unpack_lmlm(index, lmax=MMG_TABLE_LMAX):
    i1, i2 = divmod(index, (lmax+1)**2)
    return *unpack_lm(i1), *unpack_lm(i2)


def pack_MMG(l1p, l2p, lam1, lam2, l, m, lmax=MMG_TABLE_LMAX):
    return (((l1p * (lmax+1) + l2p) * (2*lmax+1) + lam1 + MMG_TABLE_LMAX) \
            * (2*lmax+1) + lam2 + MMG_TABLE_LMAX) * (2*lmax+1)**2 + pack_lm(l,m)


def unpack_MMG(index, lmax=MMG_TABLE_LMAX):
    tmp, lm = divmod(index, (2*lmax+1)**2)
    tmp, lam2 = divmod(tmp, 2*lmax+1)
    tmp, lam1 = divmod(tmp, 2*lmax+1)
    l1p, l2p = divmod(tmp, lmax+1)
    return l1p, l2p, lam1 - MMG_TABLE_LMAX, lam2 - MMG_TABLE_LMAX, *unpack_lm(lm)


def MMG_gen(fname, lmax=MMG_TABLE_LMAX):
    assert MMG_TABLE_LMAX == REAL_GAUNT_TABLE_LMAX and \
           MMG_TABLE_LMAX == REAL_ADDITION_TABLE_LMAX

    table = np.zeros(((lmax+1)**4, (lmax+1)**2 * (2*lmax+1)**4))

    for ir in range((lmax+1)**4):
        l1, m1, l2, m2 = unpack_lmlm(ir)
        M1_list = M_nz(l1, m1)
        M2_list = M_nz(l2, m2)
        for (l1p, nu1, lam1), M1 in M1_list:
            for (l2p, nu2, lam2), M2 in M2_list:
                fac = 2 * np.sqrt(np.pi / ((2*l1p+1) * (2*l2p+1)))
                G_list = real_gaunt_nz(l1p, l2p, nu1, nu2)
                for (l,m), G in G_list:
                    ic = pack_MMG(l1p, l2p, lam1, lam2, l, m)
                    table[ir, ic] += \
                            fac * np.sqrt(2*l+1) * M1 * M2 * G
        print(f'{ir+1}/{(lmax+1)**4}', end='\r')
    print('')

    table_csr = csr_matrix(table)
    save_npz(fname, table_csr)


if not os.path.isfile(MMG_TABLE):
    MMG_gen(MMG_TABLE, MMG_TABLE_LMAX)

_MMG_table = load_npz(MMG_TABLE)


def MMG_nz(l1, m1, l2, m2):
    ir = pack_lmlm(l1, m1, l2, m2)
    tab = _MMG_table[ir]
    colind = [unpack_MMG(ic) for ic in tab.indices]
    return list(zip(colind, tab.data))


def real_solid_harm_prod2(A, l1, m1, B, l2, m2, C):
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
    CA = C - A
    CB = C - B

    xpan = {}
    tab = MMG_nz(l1, m1, l2, m2)
    for (l1p, l2p, lam1, lam2, l, m), coef in tab:
        S1 = real_solid_harm(l1-l1p, lam1, CA)
        S2 = real_solid_harm(l2-l2p, lam2, CB)

        key = (l1p+l2p-l, l, m)
        val = coef * S1 * S2
        if key not in xpan:
            xpan[key] = val
        else:
            xpan[key] += val

    return xpan


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

            G_list = real_gaunt_nz(l1p, l2p, nu1, nu2)

            for (l,m), G in G_list:
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
        
        l1, m1 = 4, 2
        l2, m2 = 4, -2
        #l1, m1 = 1, 1
        #l2, m2 = 2, -2
        
        xpan = real_solid_harm_prod(A, l1, m1, B, l2, m2, C)
        #xpan = real_solid_harm_prod2(A, l1, m1, B, l2, m2, C)

        #print('xspan...')
        #for key, coef in xpan.items():
        #    print(key, coef)

        #print('xspan2...')
        #for key, coef in xpan2.items():
        #    print(key, coef)


        rCabs = np.linalg.norm(r-C)
        val = sum(coef
                  * rCabs**key[0]
                  * real_solid_harm(key[1], key[2], r-C)
                  for key, coef in xpan.items())

        ref = real_solid_harm(l1, m1, r-A) * real_solid_harm(l2, m2, r-B)
        #self.assertAlmostEqual(ref, val, 12)
        self.assertTrue(np.allclose(ref, val, rtol=1e-12))


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

        #for key, coef in xpan.items():
        #    print(key, coef)

        ref = sgto(r, A, alpha, l1, m1) * sgto(r, B, beta, l2, m2)
        self.assertAlmostEqual(ref, val, 12)


if __name__ == '__main__':
    unittest.main()


