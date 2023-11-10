import numpy as np

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


def test_ikebe():
    from scipy.special import spherical_jn

    print('Testing Ikebe\'s method...')
    
    for n in range(20):
        for nroots in range(1, 50):
            roots = ikebe(n, nroots)
            assert np.linalg.norm(spherical_jn(n, roots), np.inf) < 1e-14

    print('...Passed!')


if __name__ == '__main__':
    test_ikebe()
