import numpy as np

##########################
#       M = N-2
##########################





exit()
##########################
#       M = N-1
##########################
def triul(p, N):
    m = np.zeros((N,N))
    for i in range(N):
        m[:N-i,i] = p**i
    return m

N = 4
p = 2. / 3.0
q = 1. - p


A = np.eye(N-1) - q * triul(p, N-1)
b = (1 - p**np.arange(N-1, 0, -1)) / q

x = np.linalg.solve(A, b)
print(x[0])
print(1551/296) # N=4, p=2/3



