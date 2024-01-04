import numpy as np
import sympy as sp

class GenSol:
    '''
    Generate all non-negative integer solutions for

        \sum_k x[k] <= M

    Examples
    --------
    >>> gensol = GenSol()
    >>> gensol(2, 3)
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]

    '''
    def __init__(self):
        self.cache = {}
        self.count_cache = {}
        self.size_limit = 1000

    def __call__(self, k, M):
        num = self._count(k, M)
        print('number of solutions: %s'%num)
        if num > self.size_limit:
            print('number of solutions (%s) exceeds the limit (%s).'%(num, self.size_limit))
            exit()
        return self._gen(k, M)

    def _count(self, k, M):
        '''
        Computes the number of non-negative integer solutions for

            \sum_k x[k] <= M.

        '''
        if k == 0:
            return 0
        if (k, M) in self.count_cache:
            return self.count_cache[(k, M)]
        else:
            num = M+1 if k == 1 else sum([self._count(k-1, i) for i in range(M+1)])
            self.count_cache[(k, M)] = num
            return num

    def _gen(self, k, M):
        if k == 0:
            return []
        if (k, M) in self.cache:
            return self.cache[(k, M)]
        else:
            if k == 1:
                sol = [(i,) for i in range(M+1)]
            else:
                sol = [(i, *s) for i in range(M+1) for s in self._gen(k-1, M-i)]
            self.cache[(k, M)] = sol
            return sol

M = 3
N = 6

p = 2. / 3.
q = 1. - p

gensol = GenSol()
sol = gensol(N-M, M-1)
sz = len(sol)
index_map = { s: i for i, s in enumerate(sol)}

################################
#       numerical solution
################################
A = np.eye(sz)
b = np.zeros(sz)
for i, s in enumerate(sol):
    b[i] = ( 1 - p**(M-sum(s)) ) / q

for i, s in enumerate(sol):
    for k in range(M-sum(s)):
        A[i, index_map[(*s[1:],k)]] -= q * p**k

E = np.linalg.solve(A, b)
print(E[0])

################################
#       symbolic solution
################################

q = sp.Symbol('q')

b = sp.Matrix([(1-(1-q)**(M-sum(s)))/q for s in sol])
A = sp.eye(sz)
for i, s in enumerate(sol):
    for k in range(M-sum(s)):
        A[i, index_map[(*s[1:],k)]] -= q * (1-q)**k

x = sp.linsolve((A, b)).args[0][0]
#print(x.factor())
print(x.evalf(subs={q: 1-p}))

