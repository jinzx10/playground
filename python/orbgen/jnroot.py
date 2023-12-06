import numpy as np
from scipy.special import spherical_jn


'''
Returns the first n roots of the l-th order spherical
Bessel function of the first kind by the method of Ikebe et al.

Parameters
----------
    l : int
        Order of the spherical Bessel function.
    n : int
        Number of roots to be returned.

Returns
-------
    roots : array
        A 1-D array containing the first n roots of the l-th order spherical Bessel function.

References
----------
    Ikebe, Y., Kikuchi, Y., & Fujishiro, I. (1991).
    Computing zeros and orders of Bessel functions.
    Journal of Computational and Applied Mathematics, 38(1-3), 169-184.

'''
def ikebe(n, nroots):
    from scipy.linalg import eigvalsh_tridiagonal

    nu = n + 0.5
    sz = nroots*2 + n + 10

    alpha = nu + 2*np.arange(2, sz+1, dtype=int)

    A_diag = np.zeros(sz)
    A_diag[0] = 2. / ( (nu+3) * (nu+1) )
    A_diag[1:] = 2. / ( (alpha+1) * (alpha-1) )
    A_subdiag = 1. / ( (alpha-1) * np.sqrt(alpha*(alpha-2)) )

    eigval = eigvalsh_tridiagonal(A_diag, A_subdiag)[::-1]
    return 2. / np.sqrt(eigval[:nroots])


'''
Returns the first n roots of the l-th order spherical
Bessel function of the first kind.

The roots of j_{l} and j_{l+1} are interlaced; so are
the roots of j_{l} and j_{l+2}. This property is exploited
to bracket the roots of j_{l} by the roots of j_{l-1}
or j_{l-2} recursively.

Parameters
----------
    l : int
        Order of the spherical Bessel function.
    n : int
        Number of roots to be returned.

Returns
-------
    roots : array
        A 1-D array containing the first n roots of the l-th order spherical Bessel function.

'''
def bracket(l, nzeros):
    from scipy.optimize import brentq

    assert l >= 0 and nzeros > 0

    nz = nzeros + (l+1)//2
    buffer = np.arange(1, nz+1, dtype=float)*np.pi

    ll = 1
    jl = lambda x: spherical_jn(ll, x)

    if l % 2 == 1:
        for i in range(nz-1):
            buffer[i] = brentq(jl, buffer[i], buffer[i+1], xtol=1e-14)
        nz -= 1

    for ll in range(2 + l%2, l+1, 2):
        for i in range(nz-1):
            buffer[i] = brentq(jl, buffer[i], buffer[i+1], xtol=1e-14)
        nz -= 1

    return buffer[:nzeros]


############################################################
#                       Testing
############################################################
def test_ikebe():
    print('Testing Ikebe\'s method...')
    
    for n in range(20):
        for nroots in range(1, 50):
            roots = ikebe(n, nroots)
            assert np.linalg.norm(spherical_jn(n, roots), np.inf) < 1e-14

    print('...Passed!')


def test_bracket():
    print('Testing the bracketing method...')

    for n in range(20):
        for nroots in range(1, 50):
            roots = bracket(n, nroots)
            assert np.linalg.norm(spherical_jn(n, roots), np.inf) < 1e-14

    print('...Passed!')


if __name__ == '__main__':
    test_ikebe()
    test_bracket() # slow!

