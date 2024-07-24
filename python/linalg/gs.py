import numpy as np
import time

# classical Gram-Schmidt
def cgs(A):
    m, n = A.shape
    assert(m >= n)

    q = np.zeros((m,n))
    r = np.zeros((n,n))

    r[0,0] = np.linalg.norm(A[:,0])
    q[:,0] = A[:,0] / r[0,0]
    
    for k in range(1,n):
        r[:k,k] = q[:,:k].T @ A[:,k]
        z = A[:,k] - q[:,:k] @ r[:k,k]
        r[k,k] = np.linalg.norm(z)
        q[:,k] = z / r[k,k]

    return q, r

# modified Gram-Schmidt
def mgs(M):
    A = M.copy()
    m, n = A.shape
    assert(m >= n)

    q = np.zeros((m,n))
    r = np.zeros((n,n))

    for k in range(n):
        r[k,k] = np.linalg.norm(A[:,k])
        q[:,k] = A[:,k] / r[k,k]
        for j in range(k+1,n):
            r[k,j] = q[:,k].T @ A[:,j]
            A[:,j] -= r[k,j] * q[:,k]

    return q, r

def mgs2(A): # overwrite A with Q
    m, n = A.shape
    assert(m >= n)

    r = np.zeros((n,n))

    for k in range(n):
        r[k,k] = np.linalg.norm(A[:,k])
        A[:,k] = A[:,k] / r[k,k]
        for j in range(k+1,n):
            r[k,j] = A[:,k].T @ A[:,j]
            A[:,j] -= r[k,j] * A[:,k]


m = 10000
n = 400

M = np.random.rand(m,n)

qc, rc = cgs(M)
err_qc = np.linalg.norm(qc.T @ qc - np.eye(n))
print('cgs orthonormality error = %6.4e'%(err_qc))


#qm, rm = mgs(M)
#err_qm = np.linalg.norm(qm.T @ qm - np.eye(n))
#print('mgs orthonormality error = %6.4e'%(err_qm))

mgs2(M)
err_qm = np.linalg.norm(M.T @ M - np.eye(n))
print('mgs orthonormality error = %6.4e'%(err_qm))

#mgs_in_place(M)
#print(M)
#print(qm)



#q, r = np.linalg.qr(A)
#print(r, '\n')

#print(np.linalg.norm(q@r - A))
#print(np.linalg.norm(q1@r1 - A))
#print(np.linalg.norm(q2@r2 - A))

