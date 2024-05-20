import numpy as np

N = 6
##########################
#       M = N-2
##########################
# find all integer pairs (r,s) such that
# 0 <= r,s <= N-3 and r+s <= N-3
idx_rs = {}
idx = 0
for r in range(N-2):
    for s in range(N-2-r):
        idx_rs[(r,s)] = idx
        idx += 1
#print(idx_rs)




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



