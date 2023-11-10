import numpy as np

'''
Tail-smoothing function used in the generation of numerical radial functions.

References
----------
    Chen, M., Guo, G. C., & He, L. (2010).
    Systematically improvable optimized atomic basis sets for ab initio calculations.
    Journal of Physics: Condensed Matter, 22(44), 445501.

'''
def smoothing(r, rcut, sigma=0.1):
    if abs(sigma) < 1e-12:
        g = np.zeros_like(r)
        g[r < rcut] = 1.0
    else:
        g = 1. - np.exp(-0.5*((r-rcut)/sigma)**2)
        g[r >= rcut] = 0.0

    return g


'''
Generates a set of numerical radial functions by linear combinations of spherical Bessel functions.

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
def j2rad(coeff, q, rcut, dr=0.01, sigma=0.1):
    from scipy.integrate import simpson

    lmax = len(coeff)-1
    nzeta = [len(coeff[l]) for l in range(lmax+1)]

    nr = int(rcut/dr) + 1
    r = dr * np.arange(nr)
    g = smoothing(r, rcut, sigma)

    chi = [[np.zeros(nr)] * nzeta[l] for l in range(lmax+1)]
    for l in range(lmax+1):
        for izeta in range(nzeta[l]):
            for iq in range(len(coeff[l][izeta])):
                chi[l][izeta] += coeff[l][izeta][iq] * spherical_jn(l, q[l][izeta][iq]*r)

            # smooth & normalize
            chi[l][izeta] = chi[l][izeta] * g
            c = simpson((r*chi[l][izeta])**2, dx=dr)
            chi[l][izeta] *= 1./np.sqrt(c)

    return chi


