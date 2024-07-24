import numpy as np
from scipy.sparse import block_diag

def g(n, sz):
    for i in range(n):
        yield i * np.ones((sz, sz))

a_dense = block_diag( g(5,2) ).toarray()
a_sp = block_diag( g(5,2) )
#a = block_diag((np.eye(3), 2*np.eye(2))).toarray()
print(a_sp)
print(a_dense)

print(a_sp @ a_dense)
print(a_dense @ a_sp)

a3 = np.random.rand(3, *a_dense.shape)
print(a_sp @ a3)




