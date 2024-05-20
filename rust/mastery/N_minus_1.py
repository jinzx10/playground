# Expected number of tosses to get a mastery of M=N-1
import sympy as sp

def triul(p, N):
    m = sp.zeros(N,N)
    for i in range(N):
        for j in range(N-i):
            m[j,i] = p**i
    return m


N = sp.Integer(5)
p = sp.Rational(2,3)
q = sp.Integer(1) - p

A = sp.eye(N-1) - q * triul(p, N-1)
b = sp.Matrix([(1 - p**k) / q for k in range(N-1, 0, -1)])

x = sp.linsolve((A, b)).args[0][0]
print(x)
print(x.evalf())

#######################################
#           symbolic
#######################################
p = sp.Symbol('p')
#q = sp.Symbol('q')

A = sp.eye(N-1) - (1-p) * triul(p, N-1)
b = sp.Matrix([(1 - p**k) / (1-p) for k in range(N-1, 0, -1)])

x = sp.linsolve((A, b)).args[0][0]
print(x)
