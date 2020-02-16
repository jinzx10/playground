import numpy as np
import timeit
import scipy.linalg as sl

sz = 2000

a = np.random.rand(sz,sz)
a = a + a.transpose()

start = timeit.default_timer()

[val,vec] = np.linalg.eigh(a)

elapsed = timeit.default_timer() - start

print("numpy.linalg.eigh elapsed time = ", elapsed)



start = timeit.default_timer()

[val,vec] = sl.eigh(a)

elapsed = timeit.default_timer() - start

print("scipy.linalg.eigh elapsed time = ", elapsed)
