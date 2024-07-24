import numpy as np
import scipy
import matplotlib.pyplot as plt

sz_blk = 5
nblk = 10

S = np.zeros((sz_blk*nblk, sz_blk*nblk))

S_diag = np.eye(sz_blk)

coef = 0.05
S_offdiag = coef * np.random.randn(sz_blk, sz_blk)

for i in range(nblk):
    for j in range(nblk):
        if i == j:
            S[i*sz_blk:(i+1)*sz_blk, j*sz_blk:(j+1)*sz_blk] = S_diag
        elif i == j-1:
            S[i*sz_blk:(i+1)*sz_blk, j*sz_blk:(j+1)*sz_blk] = S_offdiag
        elif i == j+1:
            S[i*sz_blk:(i+1)*sz_blk, j*sz_blk:(j+1)*sz_blk] = S_offdiag.T



Sinv1 = np.linalg.inv(S)[sz_blk:2*sz_blk, sz_blk:2*sz_blk]

Sinv1_partial = np.linalg.inv(S[0:2*sz_blk, 0:2*sz_blk])[sz_blk:2*sz_blk, sz_blk:2*sz_blk]

print('Sinv = \n', Sinv1)
print('Sinv partial = \n', Sinv1_partial)






#plt.imshow(np.abs(S))
#plt.show()
