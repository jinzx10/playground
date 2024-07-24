import numpy as np

A = np.random.rand()
B = np.random.rand()
C = np.random.rand()
D = np.random.rand()
x0 = np.random.rand()

f = lambda x: A*(x-x0)**3 + B*(x-x0)**2 + C*(x-x0) + D

x = np.random.randn(4)
x = np.sort(x)
x1 = x[0]
x2 = x[1]
x3 = x[2]
x4 = x[3]

h1 = x2-x1
h2 = x3-x2
h3 = x4-x3

y1 = f(x1)
y2 = f(x2)
y3 = f(x3)
y4 = f(x4)


# analytic integral
I = lambda x: A/4*(x-x0)**4 + B/3*(x-x0)**3 + C/2*(x-x0)**2 + D*(x-x0)
ref = I(x4) - I(x1)

# simpson 3/8
alpha = (h1+h2+h3)/12*(2 + ((h2+h3)/h1-1) * (h3/(h1+h2)-1) )
delta = (h1+h2+h3)/12*(2 + ((h2+h1)/h3-1) * (h1/(h3+h2)-1) )
beta = (h1+h2+h3)**3/12 * (h1+h2-h3) / (h1*h2*(h2+h3))
gamma = (h1+h2+h3)**3/12 * (h3+h2-h1) / (h3*h2*(h2+h1))

simp = alpha*y1 + beta*y2 + gamma*y3 + delta*y4

print(ref, simp)




