import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr


n = 5
val = np.random.rand(n)
vec = qr(np.random.randn(n, n))[0]

A = vec @ np.diag(val) @ vec.T
A = (A + A.T) / 2

b = np.random.randn(n)

# minimize 1/2 x^T A x - b^T x

def f(x):
    return 0.5 * x @ A @ x - b @ x

def fprime(x):
    return A @ x - b

x0 = np.random.randn(n)
#print(f(x0))

# finite difference
fp_fd = np.zeros(n)
for i in range(n):
    h = 1e-5
    e = np.zeros(n)
    e[i] = h
    fp_fd[i] = (f(x0 + e) - f(x0 - e)) / (2 * h)


#print(fprime(x0))
#print(fp_fd)

res = minimize(f, x0, jac=fprime)
print(np.linalg.solve(A, b))
print(res.x)

