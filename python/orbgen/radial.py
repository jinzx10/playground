import numpy as np

'''
Tail-smoothing function used in the generation of numerical radial functions.

References
----------
    Chen, M., Guo, G. C., & He, L. (2010).
    Systematically improvable optimized atomic basis sets for ab initio calculations.
    Journal of Physics: Condensed Matter, 22(44), 445501.

'''
def smooth(r, rcut, sigma):
    if abs(sigma) < 1e-14:
        g = np.ones_like(r)
    else:
        g = 1. - np.exp(-0.5*((r-rcut)/sigma)**2)

    g[r >= rcut] = 0.0
    return g


'''
Generates wave numbers of spherical Bessel functions according to the given coeff
such that spherical_jn(n, q*r) is zero at r=rcut.
'''
def qgen(coeff, rcut):
    from jnroot import ikebe
    roots = [ikebe(l, max([len(clz) for clz in cl])) for l, cl in enumerate(coeff)]
    return [[roots[l][:len(clz)] / rcut for clz in cl] for l, cl in enumerate(coeff)]


'''
Builds a set of numerical radial functions by linear combinations of spherical Bessel functions.

Parameters
----------
    coeff : list of list of list of float
        A nested list containing the coefficients of spherical Bessel functions.
    rcut : int or float
        Cutoff radius.
    dr : float
        Grid spacing.
    sigma : float
        Smoothing parameter.
    orth : bool
        Whether to orthonormalize the radial functions.

Returns
-------
    chi : list of list of array of float
        A nested list containing the numerical radial functions.

'''
def build(coeff, rcut, dr, sigma, orth=False):
    from scipy.integrate import simpson
    from scipy.special import spherical_jn

    lmax = len(coeff)-1
    nzeta = [len(coeff[l]) for l in range(lmax+1)]

    nr = int(rcut/dr) + 1
    r = dr * np.arange(nr)

    g = smooth(r, rcut, sigma)
    q = qgen(coeff, rcut)

    chi = [[np.zeros(nr) for _ in range(nzeta[l])] for l in range(lmax+1)]
    for l in range(lmax+1):
        for zeta in range(nzeta[l]):
            for iq in range(len(coeff[l][zeta])):
                chi[l][zeta] += coeff[l][zeta][iq] * spherical_jn(l, q[l][zeta][iq]*r)

            chi[l][zeta] = chi[l][zeta] * g # apply tail-smoothing

            if orth:
                for y in range(zeta):
                    chi[l][zeta] -= simpson(r**2 * chi[l][zeta] * chi[l][y], dx=dr) * chi[l][y]

            chi[l][zeta] *= 1./np.sqrt(simpson((r*chi[l][zeta])**2, dx=dr)) # normalize

    return chi, r


############################################################
#                       Test
############################################################
import unittest

class TestRadial(unittest.TestCase):

    def test_smooth(self):
        r = np.linspace(0, 10, 100)
        rcut = 5.0

        sigma = 0.0
        g = smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[r < rcut] == 1.0) and np.all(g[r >= rcut] == 0.0))
    
        sigma = 0.5
        g = smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[r < rcut] == 1.0 - np.exp(-0.5*((r[r < rcut]-rcut)/sigma)**2)))
        self.assertTrue(np.all(g[r >= rcut] == 0.0))
    
    
    def test_qgen(self):
        from scipy.special import spherical_jn
    
        rcut = 7.0
        coeff = [[[1.0]*5, [1.0]*3], [[1.0]*7], [[1.0]*8, [1.0]*4, [1.0]*2]]
        q = qgen(coeff, rcut)
        for l, ql in enumerate(q):
            for zeta, qlz in enumerate(ql):
                self.assertEqual(len(qlz), len(coeff[l][zeta]))
                self.assertTrue(np.all(np.abs(spherical_jn(l, qlz * rcut)) < 1e-14))
    
    
    def test_build(self):
        from fileio import read_param, read_nao

        param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')

        chi, r = build(param['coeff'], param['rcut'], nao['dr'], param['sigma'], orth=True)

        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                self.assertTrue(np.all(np.abs(chi[l][zeta] - np.array(nao['chi'][l][zeta])) < 1e-12))


if __name__ == '__main__':
    unittest.main()


