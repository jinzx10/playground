import timeit
import numpy as np

n = 1000
a = np.random.rand(n,n)
b = np.random.rand(n,n)

start = timeit.default_timer()

c = a@b

elapsed = timeit.default_timer() - start

print("elapsed time = ", elapsed)

print(timeit.timeit('a@b', setup = 'import numpy as np; a = np.random.rand(1000,1000); b = np.random.rand(1000,1000)', number=100))
