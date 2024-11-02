import numpy as np

def s_becke(mu):
    p1 = 0.5 * mu * (3 - mu*mu)
    p2 = 0.5 * p1 * (3 - p1*p1)
    p3 = 0.5 * p2 * (3 - p2*p2)
    return 0.5 * (1 - p3)


def s_stratmann(mu, a=0.64):
    seg = (np.copysign(1, mu + a) - np.copysign(1, a - mu)) / 2
    x = mu / a
    x2 = x * x
    return 0.5 * (1 - seg - (1-np.abs(seg)) * \
            x * (35 + x2 * (-35 + x2 * (21 - 5 * x2))) / 16)


def u_knuth(y, b=0.8):
    core = y <= b
    edge = (not core) and (y < 1.0)
    return core + edge * 0.5 * (np.cos(np.pi * (y - b) / (1.0 - b)) + 1.0);

def s_knuth(mu, y, a=0.64, b=0.8): # modified stratmann
    return 1.0 + u_knuth(y, b) * (s_stratmann(mu, a) - 1.0)


def becke_part(drR, dRR):
    '''
    Becke partition weights.

    Given a set of atomic positions R and a point r, return the
    Becke partition weights.
    
    '''
    s = s_becke
    #s = s_stratmann
    #s = s_knuth

    nR = len(drR) 
    P = np.ones(nR)

    for I in range(nR):
        for J in range(I+1,nR):
            mu = (drR[I] - drR[J]) / (dRR[I,J])
            tmp = s(mu)
            P[I] *= tmp
            P[J] *= (1-tmp)

    #for I in range(nR):
    #    for J in range(nR):
    #        if I == J:
    #            continue
    #        mu = (drR[I] - drR[J]) / (dRR[I,J])
    #        P[I] *= s(mu)
    #        #rcut = 7
    #        #y = np.linalg.norm(r-RJ) / rcut
    #        #P[I] *= s(mu, y)

    return P / np.sum(P)



#############################################################
#                       Test
############################################################
import unittest
import matplotlib.pyplot as plt
import time

from radial import baker, murray
from delley import delley

class TestBecke(unittest.TestCase):
    pass


#def func(r, R1, R2, a1, a2):
#    return np.exp(-a1 * np.linalg.norm(r - R1.reshape((1,3)), axis=1)**2) \
#            + np.exp(-a2 * np.linalg.norm(r - R2.reshape((1,3)), axis=1)**2)

def func(r, R, a):
    '''
    Test function.

        f(r) = \sum_i exp(-a[i] * |r-R[i]|^2)

    Parameters
    ----------
    r : np.ndarray, shape=(3)
        The grid points.
    R : np.ndarray, shape=(nR,3)
        The atomic positions.
    a : np.ndarray, shape=(nR,)
        The exponents.

    '''
    val = 0.0
    for ai, Ri in zip(a, R):
        val += np.exp(-ai * np.linalg.norm(r - Ri)**2)
    return val



if __name__ == '__main__':
    #unittest.main()

    #mu = np.linspace(-1, 1, 100)
    #plt.plot(mu, s_becke(mu), label='Becke')
    #plt.plot(mu, s_stratmann(mu), label='Stratmann')
    #plt.legend()
    #plt.show()
    #exit()

    R = np.array([
        [0, 0, 0],
        [0, 0, 3],
        [0, 3, 0],
        ])

    a = np.array([1.1, 2.2, 3.3])

    nR = R.shape[0]
    print(f'nR = {nR}')

    r_rad, w_rad = baker(30, 8, 2)
    #r_rad, w_rad = murray(40, 3.0)
    r_ang, w_ang = delley(20)

    r = []
    w = []
    for irad in range(len(r_rad)):
        for iang in range(len(r_ang)):
            r.append(r_rad[irad] * np.array(r_ang[iang]))
            w.append(w_rad[irad] * w_ang[iang])

    r = np.array(r)
    w = np.array(w)
    print(f'number of grid points = {r.shape[0]}')
    print(f'radial max = {r_rad[-1]}')

    dRR = np.zeros((nR, nR))
    for I in range(nR):
        for J in range(nR):
            dRR[I,J] = np.linalg.norm(R[I] - R[J])

    elapsed = 0.0

    val = 0.0
    for iR in range(nR):
        for wi, ri in zip(w, r):
            drR = np.zeros(nR)
            for I in range(nR):
                drR[I] = np.linalg.norm(ri + R[iR] - R[I])
            start = time.time()
            w_becke = becke_part(drR, dRR)
            elapsed += time.time() - start
            val += wi * func(ri + R[iR], R, a) * w_becke[iR]

    print(f'time elapsed = {elapsed} s')

    val *= 4 * np.pi
    ref = np.sum((np.pi/a)**1.5)
    print(f'val = {val}')
    print(f'ref = {ref}')


    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(r[:,0], r[:,1], r[:,2])
    #plt.show()



