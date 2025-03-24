import os
import numpy as np

from scipy.io import savemat
from scipy.sparse import csr_matrix, save_npz, load_npz
from sympy.ntheory.multinomial import multinomial_coefficients

from harm import real_solid_harm, pack_lm, unpack_lm
from addition import M_nz, REAL_ADDITION_TABLE_LMAX
from gaunt import real_gaunt_nz, REAL_GAUNT_TABLE_LMAX

MMG_TABLE_LMAX = 4
MMG_TABLE = 'MMG_table2'
MMGSS_IMAP = 'MMGSS_imap'

np.set_printoptions(legacy='1.25')


def pack_lmlm(l1, m1, l2, m2, lmax=MMG_TABLE_LMAX):
    return pack_lm(l1, m1) * (lmax+1)**2 + pack_lm(l2, m2)


def unpack_lmlm(index, lmax=MMG_TABLE_LMAX):
    i1, i2 = divmod(index, (lmax+1)**2)
    return *unpack_lm(i1), *unpack_lm(i2)


def pack_lmlmlm(l1, m1, l2, m2, l, m, lmax=MMG_TABLE_LMAX):
    return (pack_lm(l1, m1) * (lmax+1)**2 + pack_lm(l2, m2)) * (2*lmax+1)**2 + pack_lm(l, m)


def unpack_lmlmlm(index, lmax=MMG_TABLE_LMAX):
    i12, i = divmod(index, (2*lmax+1)**2)
    return *unpack_lmlm(i12), *unpack_lm(i)


def MMG_gen(fname, lmax=MMG_TABLE_LMAX):
    assert MMG_TABLE_LMAX == REAL_GAUNT_TABLE_LMAX and \
           MMG_TABLE_LMAX == REAL_ADDITION_TABLE_LMAX

    #  (l1p,mu1,l2p,mu2) x (l1,m1,l2,m2,l,m)
    table = np.zeros(((lmax+1)**4, (lmax+1)**4 * (2*lmax+1)**2))

    for l1 in range(lmax+1):
        for m1 in range(-l1, l1+1):
            M1_list = M_nz(l1, m1)
            for l2 in range(lmax+1):
                for m2 in range(-l2, l2+1):
                    M2_list = M_nz(l2, m2)
                    for (l1p, mu1, nu1), M1 in M1_list:
                        for (l2p, mu2, nu2), M2 in M2_list:
                            ir = pack_lmlm(l1p, mu1, l2p, mu2, lmax)
                            fac = 2 * np.sqrt(np.pi / ((2*l1-2*l1p+1) * (2*l2-2*l2p+1)))
                            G_list = real_gaunt_nz(l1-l1p, l2-l2p, nu1, nu2)
                            for (l,m), G in G_list:
                                ic = pack_lmlmlm(l1, m1, l2, m2, l, m)
                                table[ir, ic] += fac * np.sqrt(2*l+1) * M1 * M2 * G


    # sum over nu1 & nu2 may introduce some cancellations
    # subject to floating-point error, so we clean them up
    table[abs(table) < 1e-12] = 0
    table_csr = csr_matrix(table)

    save_npz(fname, table_csr)
    savemat(fname,
            {
                'MMG_TABLE': table_csr,
                'MMG_TABLE_LMAX': float(lmax),
            },
            )


if not os.path.isfile(MMG_TABLE + '.npz'):
    MMG_gen(MMG_TABLE, MMG_TABLE_LMAX)

_MMG_table = load_npz(MMG_TABLE + '.npz')


def MMGSS_imap_gen(fname, lmax=MMG_TABLE_LMAX):
    '''
    Build the index mapping for contracting a dense array of

            (l1p,l2p) x (l1,m1,l2,m2,l,m)

    to

            (l1,m1,l2,m2) x (q,l,m)

    where q = (l1-l1p+l2-l2p-l)/2.

    '''
    imap = np.zeros(((lmax+1)**2, (lmax+1)**4*(2*lmax+1)**2), dtype=int)
    l1m1l2m2lm = np.arange((lmax+1)**4*(2*lmax+1)**2, dtype=int)

    l1m1l2m2, lm = np.divmod(l1m1l2m2lm, (2*lmax+1)**2)
    l1m1, l2m2 = np.divmod(l1m1l2m2, (lmax+1)**2)

    l = np.sqrt(lm).astype(int)
    l1 = np.sqrt(l1m1).astype(int)
    l2 = np.sqrt(l2m2).astype(int)

    for l1p in range(lmax+1):
        for l2p in range(lmax+1):
            ir = l1p * (lmax+1) + l2p
            q = (l1 - l1p + l2 - l2p - l) // 2
            qlm = q * (2*lmax+1)**2 + lm
            #imap[ir] = l1m1l2m2 * (lmax+1)*(2*lmax+1)**2 + qlm
            imap[ir] = l1m1l2m2 + qlm * (lmax+1)**4

    np.savez(fname, imap)
    savemat(fname,
            {
                'MMGSS_IMAP': imap,
            },
            )

if not os.path.isfile(MMG_TABLE + '.npz'):
    MMG_gen(MMG_TABLE, MMG_TABLE_LMAX)
_MMG_table = load_npz(MMG_TABLE + '.npz')

if not os.path.isfile(MMGSS_IMAP + '.npz'):
    MMGSS_imap_gen(MMGSS_IMAP, MMG_TABLE_LMAX)

_MMGSS_imap = np.load(MMGSS_IMAP + '.npz')


def MMGSS(AB):
    r'''
    Expansion of a product of two real solid harmonics.

    A product of an arbitrary pair of real solid harmonics
    can be expanded into a sum of real solid harmonics on
    an arbitrary new center:

     m1         m2        -- -- --             2q   m
    S  (r-A) * S  (r-B) = \  \  \  coef * |r-C|  * S (r-C)
     l1         l2        /  /  /                   l
                          -- -- --
                          q  l  m

    In practice this new center C is determined by the
    Gaussian product rule.

    '''
    lmax = MMG_TABLE_LMAX

    # MMG * S(l1p, mu1, AB) * S(l2p, mu2, AB)
    # (l1p,mu1,l2p,mu2) x (l1,m1,l2,m2,l,m)
    # --> (l1p, l2p) x (l1,m1,l2,m2,l,m)
    out = np.zeros(((lmax+1)**2, (lmax+1)**4 * (2*lmax+1)**2))

    # tabulate real solid harmonics in advance
    SAB = np.zeros(((lmax+1)**2, 1))
    for l in range(lmax+1):
        for m in range(-l, l+1):
            SAB[pack_lm(l, m),0] = real_solid_harm(l, m, AB)

    SS = np.kron(SAB, SAB)
    MMGSS_tmp = _MMG_table.multiply(SS).toarray()

    # sum over mu1 & mu2
    for ir, row in enumerate(MMGSS_tmp):
        l1m1, l2m2 = divmod(ir, (lmax+1)**2)
        l1p = int(np.sqrt(l1m1))
        l2p = int(np.sqrt(l2m2))
        out[l1p*(lmax+1)+l2p] += row

    return out, MMGSS_tmp, SS


def real_solid_harm_prod(MMGSS1, gamma):
    '''

    '''
    lmax = MMG_TABLE_LMAX
    fac1 = -1/(1+gamma)
    fac2 = 1/(1+1/gamma)
    tmp = np.kron(fac1**np.arange(lmax+1), fac2**np.arange(lmax+1))

    # (l1p,l2p) x (l1,m1,l2,m2,l,m)
    MMGSS2 = csr_matrix(MMGSS1 * tmp.reshape(-1, 1))

    # contract and re-index MMGSS2 to
    # (l1,m1,l2,m2) x (q, l, m)
    # where q = (l1 - l1p + l2 - l2p - l) // 2
    xpan = np.zeros(((lmax+1)**4, (lmax+1)*(2*lmax+1)**2))
    for l1p in range(lmax+1):
        for l2p in range(lmax+1):
            tab = MMGSS2[l1p*(lmax+1)+l2p]
            l1m1l2m2, lm = divmod(tab.indices, (2*lmax+1)**2)
            l = np.sqrt(lm).astype(int)
            l1m1, l2m2 = divmod(l1m1l2m2, (lmax+1)**2)
            l1 = np.sqrt(l1m1).astype(int) 
            l2 = np.sqrt(l2m2).astype(int)
            q = (l1 - l1p + l2 - l2p - l) // 2
            qlm = q * (2*lmax+1)**2 + lm
            xpan[l1m1l2m2, qlm] += tab.data

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

    def test_2(self):
        lmax = MMG_TABLE_LMAX

        AB = np.array([0.12, 0.34, 0.56])
        mmgss, mmgss_tmp, ss = MMGSS(AB)
        savemat('test.mat', {'mmgss': mmgss, 'mmgss_tmp':mmgss_tmp, 'ss':ss})
        savemat('test_MMG.mat', {'mmg': _MMG_table})

        gamma = 0.77



    def est_1(self):

        lmax = MMG_TABLE_LMAX

        r = np.random.randn(3)
        A = np.random.randn(3)
        B = np.random.randn(3)
        AB = A - B


        alpha = np.random.rand()
        beta = np.random.rand()
        gamma = alpha/beta

        tss = MMGSS(AB)
        xpan = real_solid_harm_prod(tss, gamma)


        #C = A * 0.3 + B * 0.7 # will introduce more zero coefs. why?
        C = A * alpha/(alpha+beta) + B * beta/(alpha+beta)
        rC = r - C
        rCabs = np.linalg.norm(rC)
        
        l1, m1 = 1, 1
        l2, m2 = 3, -3

        coef = xpan[pack_lmlm(l1, m1, l2, m2)]

        val = 0.0
        for i in range(len(coef)):
            q, lm = divmod(i, (2*lmax+1)**2)
            l, m = unpack_lm(lm)
            val += coef[i] * rCabs**(2*q) * real_solid_harm(l, m, rC)

        ref = real_solid_harm(l1, m1, r-A) * real_solid_harm(l2, m2, r-B)
        self.assertAlmostEqual(val, ref, 8)


    def est_real_solid_harm_prod(self):
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
        

    def est_sGTO_prod(self):

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


