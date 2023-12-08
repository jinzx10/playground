import numpy as np
from scipy.special import spherical_jn

def ikebe(n, nroots):
    '''
    Returns the first n roots of the l-th order spherical Bessel function
    by the method of Ikebe et al.
    
    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        n : int
            Number of roots to be returned.
    
    Returns
    -------
        roots : array
            The first n roots of the l-th order spherical Bessel function.
    
    References
    ----------
        Ikebe, Y., Kikuchi, Y., & Fujishiro, I. (1991).
        Computing zeros and orders of Bessel functions.
        Journal of Computational and Applied Mathematics, 38(1-3), 169-184.
    
    '''
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


def bracket(l, nzeros):
    '''
    Returns the first n roots of the l-th order spherical Bessel function
    by recursively using the bracketing method.
    
    The roots of j_{l} and j_{l+1} are interlaced; so are
    the roots of j_{l} and j_{l+2}. This property is exploited
    to bracket the roots of j_{l} by the roots of j_{l-1}
    or j_{l-2} recursively until the roots of j_{0}, which
    are known, are reached.
    
    Parameters
    ----------
        l : int
            Order of the spherical Bessel function.
        n : int
            Number of roots to be returned.
    
    Returns
    -------
        roots : array
            The first n roots of the l-th order spherical Bessel function.
    
    '''
    from scipy.optimize import brentq

    nz = nzeros + (l+1)//2
    buffer = np.arange(1, nz+1, dtype=float)*np.pi

    ll = 1
    jl = lambda x: spherical_jn(ll, x)

    # for odd  l: j_0 --> j_1 --> j_3 --> j_5 --> ... --> j_l
    # for even l: j_0 --> j_2 --> j_4 --> j_6 --> ... --> j_l

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
#                       Test
############################################################
import unittest

class TestJnRoot(unittest.TestCase):
    def test_ikebe(self): # fast!
        for n in range(20):
            for nroots in range(1, 50):
                roots = ikebe(n, nroots)
                self.assertLess(np.linalg.norm(spherical_jn(n, roots), np.inf), 1e-14)
    
    
    def test_bracket(self): # slow!
        for n in range(20):
            for nroots in range(1, 50):
                roots = bracket(n, nroots)
                self.assertLess(np.linalg.norm(spherical_jn(n, roots), np.inf), 1e-14)


if __name__ == '__main__':
    unittest.main()

