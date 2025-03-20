import os
import numpy as np

from scipy.io import savemat
from scipy.sparse import csr_matrix, save_npz, load_npz
from sympy.ntheory.multinomial import multinomial_coefficients

from harm import real_solid_harm, pack_lm, unpack_lm
from addition import M_nz, REAL_ADDITION_TABLE_LMAX
from gaunt import real_gaunt_nz, REAL_GAUNT_TABLE_LMAX

MMG_TABLE_LMAX = 4
MMG_TABLE = './MMG_table2.npz'

np.set_printoptions(legacy='1.25')


def pack_lmlm(l1, m1, l2, m2, lmax=MMG_TABLE_LMAX):
    return pack_lm(l1, m1) * (lmax+1)**2 + pack_lm(l2, m2)


def unpack_lmlm(index, lmax=MMG_TABLE_LMAX):
    i1, i2 = divmod(index, (lmax+1)**2)
    return *unpack_lm(i1), *unpack_lm(i2)


def pack_MMG(l1p, l2p, lam1, lam2, l, m, lmax=MMG_TABLE_LMAX):
    return (( (l1p * (lmax+1) + l2p) \
            * (2*lmax+1) + lam1 + MMG_TABLE_LMAX) \
            * (2*lmax+1) + lam2 + MMG_TABLE_LMAX) \
            * (2*lmax+1)**2 + pack_lm(l,m)


def unpack_MMG(index, lmax=MMG_TABLE_LMAX):
    tmp, lm = divmod(index, (2*lmax+1)**2)
    tmp, lam2 = divmod(tmp, 2*lmax+1)
    tmp, lam1 = divmod(tmp, 2*lmax+1)
    l1p, l2p = divmod(tmp, lmax+1)
    return l1p, l2p, lam1 - MMG_TABLE_LMAX, lam2 - MMG_TABLE_LMAX, \
            *unpack_lm(lm)


def MMG_gen(fname, lmax=MMG_TABLE_LMAX):
    assert MMG_TABLE_LMAX == REAL_GAUNT_TABLE_LMAX and \
           MMG_TABLE_LMAX == REAL_ADDITION_TABLE_LMAX

    table = np.zeros(((lmax+1)**4 * (2*lmax+1)**2, (lmax+1)**4))

    for l1 in range(lmax+1):
        for m1 in range(-l1, l1+1):
            M1_list = M_nz(l1, m1)
            for l2 in range(lmax+1):
                for m2 in range(-l2, l2+1):
                    M2_list = M_nz(l2, m2)
                    for (l1p, nu1, lam1), M1 in M1_list:
                        for (l2p, nu2, lam2), M2 in M2_list:
                            ic = 
                            fac = 2 * np.sqrt(np.pi / ((2*l1-2*l1p+1) * (2*l2-2*l2p+1)))
                            G_list = real_gaunt_nz(l1-l1p, l2-l2p, lam1, lam2)
                            for (l,m), G in G_list:
                                ir = 
                                table[ir, ic] += fac * np.sqrt(2*l+1) * M1 * M2 * G


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

    # sum over nu1 & nu2 may introduce some cancellations
    # subject to floating-point error, so we clean them up
    table[abs(table) < 1e-15] = 0

    table_csr = csr_matrix(table)
    save_npz(fname, table_csr)

    # MATLAB uses CSC format, so it's better to transpose
    savemat(fname.replace('.npz', '.mat'),
            {'MMG_TABLE': table_csr.transpose(),
             'MMG_TABLE_LMAX': float(lmax)})


if not os.path.isfile(MMG_TABLE):
    MMG_gen(MMG_TABLE, MMG_TABLE_LMAX)

_MMG_table = load_npz(MMG_TABLE)


def MMG2_nz(l1, m1, l2, m2):
    ir = pack_lmlm(l1, m1, l2, m2)
    tab = _MMG_table[ir]
    colind = [unpack_MMG(ic) for ic in tab.indices]
    return list(zip(colind, tab.data))


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
    CA = C - A
    CB = C - B

    xpan = {}
    tab = MMG_nz(l1, m1, l2, m2)
    for (l1p, l2p, lam1, lam2, l, m), coef_tab in tab:
        S1 = real_solid_harm(l1-l1p, lam1, CA)
        S2 = real_solid_harm(l2-l2p, lam2, CB)

        # express |r|**(l1p+l2p-l) as x**k1 * y**k2 * z**k3
        q = (l1p + l2p - l) // 2
        multinom = multinomial_coefficients(3, q)

        for (k1, k2, k3), coef_nom in multinom.items():
            key = (k1, k2, k3, l, m)
            val = coef_nom * coef_tab * S1 * S2
            if key not in xpan:
                xpan[key] = val
            else:
                xpan[key] += val

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

        #C = A * 0.3 + B * 0.7 # will introduce more zero coefs. why?
        C = A * 0.22 + B * 0.78
        rC = r - C
        xC, yC, zC = rC

        l1, m1 = 4, 2
        l2, m2 = 4, 4
        
        xpan = real_solid_harm_prod(A, l1, m1, B, l2, m2, C)

        val = sum(coef
                  * xC**(2*k1) * yC**(2*k2) * zC**(2*k3) 
                  * real_solid_harm(l, m, rC)
                  for (k1, k2, k3, l, m), coef in xpan.items())

        #count = 0
        #for (k1, k2, k3, l, m), coef in xpan.items():
        #    print(f'k1={k1}  k2={k2}  k3={k3}  l={l}  m={m}  coef={coef}')
        #    if abs(coef) > 1e-12:
        #        count += 1

        #print(len(xpan), count)

        ref = real_solid_harm(l1, m1, r-A) * real_solid_harm(l2, m2, r-B)
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
        xC, yC, zC = rC

        l1, m1 = 4, 2
        l2, m2 = 4, -2

        xpan = sGTO_prod(alpha, A, l1, m1, beta, B, l2, m2)
        val = sum(coef
                  * xC**(2*k1) * yC**(2*k2) * zC**(2*k3)
                  * sgto(r, C, gamma, l, m)
                  for (k1, k2, k3, l, m), coef in xpan.items())

        ref = sgto(r, A, alpha, l1, m1) * sgto(r, B, beta, l2, m2)
        self.assertAlmostEqual(ref, val, 12)


if __name__ == '__main__':
    unittest.main()


