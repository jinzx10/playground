import numpy as np
from prod import sGTO_prod
from harm import pack_lm, unpack_lm 
from prod import MMG_TABLE_LMAX

lmax = MMG_TABLE_LMAX 

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

for (l1, m1, l2, m2), (n, nnz) in count.items():
    print(f'l1={l1}  m1={m1:2}  l2={l2}  m2={m2:2}  '
          f'n={n:3}  nnz={nnz:3}')

