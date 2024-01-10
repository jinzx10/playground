import numpy as np

def _smooth(r, rcut, sigma):
    '''
    Smoothing function used in the generation of numerical radial functions.

    Parameters
    ----------
        r : array of float
            Radial grid.
        rcut : int or float
            Cutoff radius.
        sigma : float
            Smoothing parameter.

    Returns
    -------
        g : array of float
            Smoothing function on the radial grid.
    
    References
    ----------
        Chen, M., Guo, G. C., & He, L. (2010).
        Systematically improvable optimized atomic basis sets for ab initio calculations.
        Journal of Physics: Condensed Matter, 22(44), 445501.
    
    '''
    if abs(sigma) < 1e-14:
        g = np.ones_like(r)
    else:
        g = 1. - np.exp(-0.5*((r-rcut)/sigma)**2)

    g[r >= rcut] = 0.0
    return g


def _qgen(coeff, rcut):
    '''
    Wave numbers of spherical Bessel functions.

    This function generates wave numbers for spherical Bessel functions according to
    the given cutoff radius and coeff such that spherical_jn(l, coeff[l]*rcut) is zero.

    Parameters
    ----------
        coeff : list of list of list of float
            A nested list containing the coefficients of spherical Bessel functions.
        rcut : int or float
            Cutoff radius.

    Returns
    -------
        q : list of list of list of float
            A nested list containing the wave numbers of spherical Bessel functions.

    '''
    from jlzeros import ikebe
    zeros = [ikebe(l, max([len(clz) for clz in cl])) for l, cl in enumerate(coeff)]
    return [[zeros[l][:len(clz)] / rcut for clz in cl] for l, cl in enumerate(coeff)]


def build(coeff, rcut, dr, sigma, orth=False):
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
            Whether to orthogonalize the radial functions.
    
    Returns
    -------
        chi : list of list of array of float
            A nested list containing the numerical radial functions.

        r : array of float
            Radial grid.
    
    '''
    from scipy.integrate import simpson
    from scipy.special import spherical_jn

    lmax = len(coeff)-1
    nzeta = [len(coeff[l]) for l in range(lmax+1)]

    nr = int(rcut/dr) + 1
    r = dr * np.arange(nr)

    g = _smooth(r, rcut, sigma)
    q = _qgen(coeff, rcut)

    chi = [[np.zeros(nr) for _ in range(nzeta[l])] for l in range(lmax+1)]
    for l in range(lmax+1):
        for zeta in range(nzeta[l]):
            for iq in range(len(coeff[l][zeta])):
                chi[l][zeta] += coeff[l][zeta][iq] * spherical_jn(l, q[l][zeta][iq]*r)

            chi[l][zeta] = chi[l][zeta] * g # smooth the tail

            if orth:
                for y in range(zeta):
                    chi[l][zeta] -= simpson(r**2 * chi[l][zeta] * chi[l][y], dx=dr) * chi[l][y]

            chi[l][zeta] *= 1./np.sqrt(simpson((r*chi[l][zeta])**2, dx=dr)) # normalize

    return chi, r


############################################################
#                       Test
############################################################
import unittest

class _TestRadial(unittest.TestCase):

    def test_smooth(self):
        r = np.linspace(0, 10, 100)
        rcut = 5.0

        sigma = 0.0
        g = _smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[r < rcut] == 1.0) and np.all(g[r >= rcut] == 0.0))
    
        sigma = 0.5
        g = _smooth(r, rcut, sigma)
        self.assertTrue(np.all(g[r < rcut] == 1.0 - np.exp(-0.5*((r[r < rcut]-rcut)/sigma)**2)))
        self.assertTrue(np.all(g[r >= rcut] == 0.0))
    
    
    def test_qgen(self):
        from scipy.special import spherical_jn
    
        rcut = 7.0
        coeff = [[[1.0]*5, [1.0]*3], [[1.0]*7], [[1.0]*8, [1.0]*4, [1.0]*2]]
        q = _qgen(coeff, rcut)
        for l, ql in enumerate(q):
            for zeta, qlz in enumerate(ql):
                self.assertEqual(len(qlz), len(coeff[l][zeta]))
                self.assertTrue(np.all(np.abs(spherical_jn(l, qlz * rcut)) < 1e-14))
    
    
    def test_build(self):
        from scipy.integrate import simpson
        from fileio import read_param, read_nao

        param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        nao = read_nao('./testfiles/In_gga_10au_100Ry_3s3p3d2f.orb')
        chi, r = build(param['coeff'], param['rcut'], nao['dr'], param['sigma'], orth=True)

        # check normalization
        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                self.assertAlmostEqual(simpson((r*chi[l][zeta])**2, dx=nao['dr']), 1.0, places=12)

        # check orthogonality
        for l in range(len(chi)):
            for zeta in range(1, len(chi[l])):
                for y in range(zeta):
                    self.assertAlmostEqual(simpson(r**2 * chi[l][zeta] * chi[l][y], dx=nao['dr']), 0.0, places=12)

        # cross check with NAO file
        for l in range(len(chi)):
            for zeta in range(len(chi[l])):
                self.assertTrue(np.all(np.abs(chi[l][zeta] - np.array(nao['chi'][l][zeta])) < 1e-12))


if __name__ == '__main__':
    unittest.main()


