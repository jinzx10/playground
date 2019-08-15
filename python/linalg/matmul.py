import timeit
from numpy import random

n = 1000
a = random.rand(n,n)
b = random.rand(n,n)

start = timeit.default_timer()

c = a@b

elapsed = timeit.default_timer() - start

print("elapsed time = ", elapsed)
