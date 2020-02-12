import numpy as np
import timeit

sz = 4000

start = timeit.default_timer()

a = np.random.rand(sz,sz)
a = a + a.transpose()
[val,vec] = np.linalg.eig(a)

elapsed = timeit.default_timer() - start

print("elapsed time = ", elapsed)
