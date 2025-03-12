import numpy as np
from prod import sGTO_prod
from harm import pack_lm, unpack_lm 
from gaunt import REAL_GAUNT_TABLE_LMAX

lmax = REAL_GAUNT_TABLE_LMAX

A = np.random.randn(3)
B = np.random.randn(3)
alpha = np.random.rand()
beta  = np.random.rand()

count = {}
for i1 in range((lmax+1)**2):
    l1, m1 = unpack_lm(i1)
    for i2 in range(i1, (lmax+1)**2):
        l2, m2 = unpack_lm(i2)
        xpan = sGTO_prod(alpha, A, l1, m1, beta, B, l2, m2)
        xpan_nz = {k: v for k, v in xpan.items() if abs(v) > 1e-12}

        key = (l1, m1, l2, m2)
        count[key] = (len(xpan), len(xpan_nz))


for key, val in count.items():
    print(key, val)

