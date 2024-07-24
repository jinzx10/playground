import numpy as np
from scipy.linalg import solve, lstsq

'''Lowdin orthogonalization

This script compares two implementations of the Lowdin orthogonalization.

'''
def lowdin_invsqrt(A):
    '''
    Lowdin orthogonalization using the inverse square root of the overlap matrix.

    Parameters
    ----------
    A : 2D ndarray
        Column-wise (non-orthonormal) basis.


    Returns
    -------
    A_orth : 2D ndarray
        Column-wise orthonormal basis.

    '''
    w, v = np.linalg.eigh(A.T.conj() @ A)
    return A @ v @ np.diag(1.0 / np.sqrt(w)) @ v.T.conj()

    #sqrtS = v @ np.diag(np.sqrt(w)) @ v.T.conj()
    #return lstsq(sqrtS.T, A.T)[0].T


def lowdin_svd(A):
    '''
    Lowdin orthogonalization using the singular value decomposition.

    Parameters
    ----------
    A : 2D ndarray
        Column-wise (non-orthonormal) basis.


    Returns
    -------
    A_orth : 2D ndarray
        Column-wise orthonormal basis.

    '''
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    return u @ vh

def lowdin_S_svd(S):
    '''
    Lowdin orthogonalization using the singular value decomposition.

    Parameters
    ----------
    S : 2D ndarray
            


    Returns
    -------
    A_orth : 2D ndarray
        Column-wise orthonormal basis.

    '''
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    return u @ vh


if __name__ == '__main__':

    sz = 2000
    n = 20

    A = np.random.randn(sz, n) + 1j * np.random.randn(sz, n)

    # the last column is almost a linear combination of the others
    fac = 0.0001
    A[:,-1] = A[:,:-1] @ np.random.randn(n-1) + fac * A[:,-1]

    # column-wise normalization
    A /= np.linalg.norm(A, axis=0)

    orth_dev = lambda a: np.linalg.norm(a.T.conj() @ a - np.eye(a.shape[1]))

    A_invsqrt = lowdin_invsqrt(A)
    A_svd = lowdin_svd(A)

    print('difference between invsqrt and svd: ', np.linalg.norm(A_invsqrt - A_svd, np.inf))

    # transform matrix from A to A_svd
    # A @ T = A_svd --> A_svd.H @ A @ T = I --> T = inv(A_svd.H @ A)
    T = np.linalg.inv(A_svd.T.conj() @ A)

    iT = A_svd.T.conj() @ A

    A_svd2 = A @ T
    A_svd3 = np.linalg.solve(iT.T, A.T).T
    print('linear transformation error:', np.linalg.norm(A_svd - A_svd2, np.inf))
    print('linear transformation error:', np.linalg.norm(A_svd - A_svd3, np.inf))

    print('input: orth dev: ', orth_dev(A))
    print('invsqrt orth dev: ', orth_dev(A_invsqrt))
    print('svd orth dev: ', orth_dev(A_svd))



