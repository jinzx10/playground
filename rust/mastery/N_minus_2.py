import numpy as np

class NumSol:
    '''
    Counts the number of non-negative integer solutions for

        \sum_k x[k] <= M

    Examples
    --------
    >>> num_sol = NumSol()
    >>> num_sol(2, 3)
    10

    The solutions are
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
    (1, 1), (1, 2), (2, 0), (2, 1), (3, 0).

    '''
    def __init__(self):
        self.k = 0
        self.M = 0
        self.cache = {}

    def __call__(self, k, M):
        if (k, M) in self.cache:
            return self.cache[(k, M)]
        else:
            num = M+1 if k == 1 else sum([self(k-1, i) for i in range(M+1)])
            self.cache[(k, M)] = num
            return num

class GenSolve:
    '''
    Generate all non-negative integer solutions for

        \sum_k x[k] <= M

    Examples
    --------
    >>> gensol = GenSolve()
    >>> gensol(2, 3)
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)]

    '''
    def __init__(self):
        self.k = 0
        self.M = 0
        self.cache = {}

    def __call__(self, k, M):
        if (k, M) in self.cache:
            return self.cache[(k, M)]
        else:
            if k == 1:
                sol = [(i,) for i in range(M+1)]
            else:
                sol = [(i, *s) for i in range(M+1) for s in self(k-1, M-i)]
            self.cache[(k, M)] = sol
            return sol

M = 6
N = 8

p = 2./3.
q = 1. - p

num_sol = NumSol()

sol = gensol(N-M, M-1)

glb2loc = { s: i for i, s in enumerate(sol)}
sz = len(sol)

A = np.eye(sz)
b = np.zeros(sz)
for i, (r,s) in enumerate(sol):
    b[i] = (1-p**(N-2-r-s))/q

for i, (r,s) in enumerate(sol):
    for k in range(N-2-r-s):
        A[i, glb2loc[(s,k)]] -= q * p**k

E = np.linalg.solve(A, b)
print(E[0])


