import numpy as np

def s_becke(mu):
    p1 = 0.5 * mu * (3 - mu*mu)
    p2 = 0.5 * p1 * (3 - p1*p1)
    p3 = 0.5 * p2 * (3 - p2*p2)
    return 0.5 * (1 - p3)


def s_stratmann(mu, a=0.64):
    x = mu / a;
    x2 = x * x;
    h = 0.0625 * x * (35 + x2 * (-35 + x2 * (21 - 5 * x2)));
    mid = abs(x) < 1;
    g = (not mid) * np.sign(x) + mid * h;
    return 0.5 * (1.0 - g);


def u_knuth(y, b=0.8):
    core = y <= b
    edge = (not core) and (y < 1.0)
    return core + edge * 0.5 * (np.cos(np.pi * (y - b) / (1.0 - b)) + 1.0);


def s_knuth(mu, y, a=0.64, b=0.8): # modified stratmann
    return 1.0 + u_knuth(y, b) * (s_stratmann(mu, a) - 1.0)


def becke(drR, dRR, iR, c):
    '''
    Becke partition weights.

    Parameters
    ----------
        drR : array
            Distance between the grid point and centers.
        dRR : array
            Cistance between centers.
        iR : array
            Indices of relevant centers. The partition weight is
            computed only for these centers, not all centers.
        c : int
            The index of the center whom the grid point belongs to.

    '''
    nR = len(iR)
    P = np.ones(nR)
    for i in range(nR):
        I = iR[i]
        for j in range(i+1, nR):
            J = iR[j]
            mu = (drR[I] - drR[J]) / dRR[I,J]
            s = s_becke(mu)
            P[I] *= s
            P[J] *= (1.0 - s)

    return P[c] / np.sum(P)


def stratmann0(drR, dRR, iR, c):
    '''
    Stratmann partition weights (computed without screening).

    Parameters
    ----------
        drR : array
            Distance between the grid point and centers.
        dRR : array
            Cistance between centers.
        iR : array
            Indices of relevant centers. The partition weight is
            computed only for these centers, not all centers.
        c : int
            The index of the center whom the grid point belongs to.

    '''
    nR = len(iR)
    P = np.ones(nR)
    for i in range(nR):
        I = iR[i]
        for j in range(i+1, nR):
            J = iR[j]
            mu = (drR[I] - drR[J]) / dRR[I,J]
            s = s_stratmann(mu)
            P[I] *= s
            P[J] *= (1.0 - s)

    return P[c] / np.sum(P)


def f_mod(x, Rc, c=0.6):
    if x <= c*Rc:
        return x
    elif x < Rc:
        return x + np.exp(-(Rc-c*Rc) / (x-c*Rc)) / (Rc - x)
    else:
        return 1e10


def stratmann_mod1(drR, dRR, iR, c, Rcut):
    '''
    My modified stratmann partition weights (computed without screening).

    Parameters
    ----------
        drR : array
            Distance between the grid point and centers.
        dRR : array
            Cistance between centers.
        iR : array
            Indices of relevant centers. The partition weight is
            computed only for these centers, not all centers.
        c : int
            The index of the center whom the grid point belongs to.

    '''
    nR = len(iR)
    P = np.ones(nR)
    for i in range(nR):
        I = iR[i]
        for j in range(i+1, nR):
            J = iR[j]
            #mu = (drR[I] - drR[J]) / dRR[I,J]
            mu = (f_mod(drR[I], Rcut[I]) - f_mod(drR[J], Rcut[J])) / dRR[I,J]
            s = s_stratmann(mu)
            P[I] *= s
            P[J] *= (1.0 - s)

    return P[c] / np.sum(P)


def stratmann(drR, dRR, drR_thr, iR, c):
    '''
    Stratmann partition weights.

    '''
    # If r falls within the exclusive zone of a center, return immediately.
    I = iR[c]
    for J in iR:
        if drR[J] <= drR_thr[J]:
            return float(I == J)

    # Even if the grid point does not fall within the exclusive zone of any
    # center, the normalized weight could still be 0 or 1, and this can be
    # figured out by examining the unnormalized weight alone.

    # Swap the grid center to the first position in iteration for convenience.
    # Restore the original order before return.
    iR[0], iR[c] = iR[c], iR[0]

    nR = len(iR)
    P = np.ones(nR)
    for j in range(1, nR):
        J = iR[j]
        mu = (drR[I] - drR[J]) / dRR[I,J]
        P[j] = s_stratmann(mu)

    P[0] = np.prod(P[1:])
    if P[0] == 0.0 or P[0] == 1.0:
        iR[0], iR[c] = iR[c], iR[0]
        return P[0]

    # If it passes all the screening above, all unnormalized weights
    # have to be calculated in order to get the normalized weight.
    P[1:] = 1.0 - P[1:]
    for i in range(1, nR):
        I = iR[i]
        for j in range(i+1, nR):
            J = iR[j]
            mu = (drR[I] - drR[J]) / dRR[I,J]
            s = s_stratmann(mu)
            P[i] *= s
            P[j] *= (1.0 - s)

    iR[0], iR[c] = iR[c], iR[0]
    return P[0] / np.sum(P)


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

def func2(r, R, Rcut, c):
    '''
    Test function.

               /
        f(r) = | \sum_i (1 + cos( pi * |r-R[i]| / (c*Rcut[i]) ))     |r-R[i]| < c*Rcut[i]
               |
               \     0       |r-R[i]| >= c*Rcut[i]

    Parameters
    ----------
    r : np.ndarray, shape=(3)
        The grid points.
    R : np.ndarray, shape=(nR,3)
        The atomic positions.
    Rcut : np.ndarray, shape=(nR,)
        Cutoff radii.

    '''
    val = 0.0
    for Ri, Rc in zip(R, Rcut):
        d = np.linalg.norm(r - Ri)
        if d < c*Rc:
            val += (1.0 + np.cos(np.pi * d / (c*Rc)))
    return val


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

    #n = 1000
    #x = np.linspace(0, 0.8, n)
    #d = np.zeros(n)
    #for i, xi in enumerate(x):
    #    d[i] = abs(s_stratmann(xi) + s_stratmann(-xi) - 1)

    #print(f'{np.max(d):8.2e}')

    #exit()

    R = np.array([
        [0, 0, 0],
        [0, 0, 3],
        [0, 3, 0],
        ])

    a = np.array([1.1, 2.2, 3.3])

    nR = R.shape[0]
    print(f'nR = {nR}')

    r_rad, w_rad = baker(50, 8, 1)
    r_ang, w_ang = delley(17)

    print(r_rad)

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

    # radii of exclusive zones
    drR_thr = np.zeros(nR)
    for I in range(nR):
        dRRmin = 1e100
        for J in range(nR):
            if J != I:
                dRRmin = min(dRRmin, dRR[I,J])
        drR_thr[I] = 0.18 * dRRmin

    # cutoff radii
    Rcut = np.ones(nR) * r_rad[-1]
    print(Rcut)
    c = 0.4

    elapsed = 0.0
    val = 0.0
    for iR in range(nR):
        for wi, ri in zip(w, r):
            drR = np.zeros(nR)
            for I in range(nR):
                drR[I] = np.linalg.norm(ri + R[iR] - R[I])

            start = time.time()
            #w_part = becke(drR, dRR, range(nR), iR)
            #w_part = stratmann0(drR, dRR, range(nR), iR)
            w_part = stratmann_mod1(drR, dRR, np.arange(nR), iR, Rcut)
            #w_part = stratmann(drR, dRR, drR_thr, np.arange(nR), iR)
            elapsed += time.time() - start

            #val += wi * func(ri + R[iR], R, a) * w_part
            val += wi * func2(ri + R[iR], R, Rcut, c) * w_part

    print(f'time elapsed = {elapsed} s')

    val *= 4 * np.pi
    #ref = np.sum((np.pi/a)**1.5)
    ref = np.sum( (c*Rcut)**3 * (1.0/3 - 2 / np.pi**2) ) * 4 * np.pi
    print(f'val = {val}')
    print(f'ref = {ref}')


    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(r[:,0], r[:,1], r[:,2])
    #plt.show()



