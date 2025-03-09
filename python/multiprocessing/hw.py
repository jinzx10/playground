import numpy as np
import time

import jax.numpy as jnp

np.random.seed(0)

class Test:

    def __init__(self, sz, n):
        self.M = [np.linalg.qr(np.random.randn(sz, sz))[0] for _ in range(n)]
        self.M = [jnp.array(m) for m in self.M]


    def mul(self, i):
        return (self.M[i] @ self.M[i].T @ self.M[i] @ self.M[i].T).sum()

    def sqr(self):
        n = len(self.M)

        start = time.time()
        M_sqr = [self.mul(i) for i in range(n)]
        print('serial')
        print('Time: ', time.time() - start)
        print('Sum : ', sum(M_sqr))
        print('')


    def sqr_mp(self, cpus):
        import multiprocessing as mp

        n = len(self.M)

        pool = mp.Pool(processes=cpus)
        start = time.time()
        M_sqr = pool.map(self.mul, range(n))
        print('mp  : ', cpus)
        print('Time: ', time.time() - start)
        print('Sum : ', sum(M_sqr))
        print('')

    def sqr_mt(self, threads):
        from multiprocessing.pool import ThreadPool
        n = len(self.M)

        pool = ThreadPool(threads)

        start = time.time()
        M_sqr = pool.map(self.mul, range(n))
        print('mt  : ', threads)
        print('Time: ', time.time() - start)
        print('Sum : ', M_sqr)
        print('')


t = Test(2000, 16)
t.sqr()
t.sqr_mt(1)
t.sqr_mt(4)
#t.sqr_mp(1)
#t.sqr_mp(8)

