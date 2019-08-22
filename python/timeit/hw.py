import numpy as np
import timeit

n = 2000

a = np.random.rand(n,n)
b = np.random.rand(n,n)

start = timeit.default_timer()
c = a@b
elapsed = timeit.default_timer() - start
print(elapsed)

start = timeit.default_timer()
c = a*b
elapsed = timeit.default_timer() - start

print(elapsed)
