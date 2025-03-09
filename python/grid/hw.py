import numpy as np

def eigh3(A):
    '''
    Given a real symmetric 3x3 matrix A, compute the eigenvalues
    Note that acos and cos operate on angles in radians

    '''

    p1 = A[0,1]**2 + A[0,2]**2 + A[1,2]**2
    if p1 == 0:
        # A is diagonal
        return np.diag(A)
    else:
        q = np.trace(A) / 3
        p2 = (A[0,0] - q)**2 + (A[1,1] - q)**2 + (A[2,2] - q)**2 + 2 * p1
        p = np.sqrt(p2 / 6)
        B = (1. / p) * (A - q * np.eye(3))
        r = np.linalg.det(B) / 2
    
        # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        # but computation error can leave it slightly outside this range.
        if r <= -1:
            phi = pi / 3
        elif r >= 1:
            phi = 0
        else:
            phi = np.arccos(r) / 3
    
        # the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1 = q + 2 * p * np.cos(phi)
        eig3 = q + 2 * p * np.cos(phi + (2*np.pi/3))
        eig2 = 3 * q - eig1 - eig3
        return np.array([eig3, eig2, eig1])


a = np.random.randn(3, 3)
a = a + a.T

val, vec = np.linalg.eigh(a)
#print(f'max eigenvalue: {val[-1]}')
print(f'Eigenvalues: {val}')
print(f'Eigenvalues: {eigh3(a)}')



