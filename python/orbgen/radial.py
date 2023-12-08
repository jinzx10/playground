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
    q : list of list of list of float
        Wave numbers of each spherical Bessel component.
    rcut : int or float
        Cutoff radius.
    dr : float
        Grid spacing.
    sigma : float
        Smoothing parameter.

Returns
-------
    chi : list of list of array of float
        A nested list containing the numerical radial functions.

'''
def build(coeff, rcut, dr, sigma, q=None):
    from scipy.integrate import simpson
    from scipy.special import spherical_jn

    if q is None:
        q = qgen(coeff, rcut)

    lmax = len(coeff)-1
    nzeta = [len(coeff[l]) for l in range(lmax+1)]

    nr = int(rcut/dr) + 1
    r = dr * np.arange(nr)
    g = smoothing(r, rcut, sigma)

    chi = [[np.zeros(nr) for _ in range(nzeta[l])] for l in range(lmax+1)]
    for l in range(lmax+1):
        for izeta in range(nzeta[l]):
            for iq in range(len(coeff[l][izeta])):
                chi[l][izeta] += coeff[l][izeta][iq] * spherical_jn(l, q[l][izeta][iq]*r)

            chi[l][izeta] = chi[l][izeta] * g # apply tail-smoothing
            chi[l][izeta] *= 1./np.sqrt(simpson((r*chi[l][izeta])**2, dx=dr)) # normalize

    return chi, r


############################################################
#                       Testing
############################################################
def test_smooth():
    print('Testing smooth...')

    r = np.linspace(0, 10, 100)
    rcut = 5.0
    sigma = 0.0
    g = smooth(r, rcut, sigma)
    assert np.all(g[r < rcut] == 1.0) and np.all(g[r >= rcut] == 0.0)

    sigma = 0.5
    g = smooth(r, rcut, sigma)
    assert np.all(g[r < rcut] == 1.0 - np.exp(-0.5*((r[r < rcut]-rcut)/sigma)**2))
    assert np.all(g[r >= rcut] == 0.0)

    print('...Passed!')


def test_qgen():
    print('Testing qgen...')
    from scipy.special import spherical_jn

    rcut = 7.0
    coeff = [[[1.0]*5, [1.0]*3], [[1.0]*7], [[1.0]*8, [1.0]*4, [1.0]*2]]
    q = qgen(coeff, rcut)
    for l, ql in enumerate(q):
        for izeta, qlz in enumerate(ql):
            assert len(qlz) == len(coeff[l][izeta])
            assert np.all(np.abs(spherical_jn(l, qlz * rcut)) < 1e-14)

    print('...Passed!')


def test_build():
    pass

if __name__ == '__main__':
    test_smooth()
    test_qgen()
    test_build()


