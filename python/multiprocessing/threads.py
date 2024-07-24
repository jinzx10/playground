from multiprocessing.pool import ThreadPool

nthreads = 4
pool = ThreadPool(nthreads)

def f(x):
    return x*x


xsqr = pool.map(f, range(10))
print(xsqr)
