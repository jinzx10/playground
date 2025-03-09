import numpy as np


def divide(grids):
    '''

    '''
    center = np.mean(grids, axis=0)
    print(center)
    d = grids - center
    val, vec = np.linalg.eigh(np.dot(d.T, d))
    return vec[:, -1]


r1 = np.array([0, 0, 0])
r2 = np.array([2, 2, -2])

ngrid1 = 100
ngrid2 = 100

dr1 = np.random.randn(ngrid1, 3)
dr2 = np.random.randn(ngrid2, 3)

grid1 = r1 + dr1
grid2 = r2 + dr2
grids = np.concatenate((grid1, grid2), axis=0)

print(grids.shape)

print(divide(grids))

