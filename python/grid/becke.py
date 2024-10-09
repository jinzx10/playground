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


def s_knuth(mu, Rc, a=0.64, b=0.8): # modified stratmann
    pass


def becke_part(R, r):
    '''
    Becke partition weights.

    Given a set of atomic positions R and a point r, return the
    Becke partition weights.
    
    '''
    #s = s_becke
    s = s_stratmann
    P = np.ones(len(R))
    for I, RI in enumerate(R):
        for J, RJ in enumerate(R):
            if I == J:
                continue
            mu = (np.linalg.norm(r-RI) - np.linalg.norm(r-RJ)) \
                    / (np.linalg.norm(RI-RJ))
            P[I] *= s(mu)

    return P / np.sum(P)



#############################################################
#                       Test
############################################################
import unittest
import matplotlib.pyplot as plt

from radial import baker
from delley import delley

class TestBecke(unittest.TestCase):
    pass


def func(r, R1, R2, a1, a2):
    return np.exp(-a1 * np.linalg.norm(r - R1.reshape((1,3)), axis=1)**2) \
            + np.exp(-a2 * np.linalg.norm(r - R2.reshape((1,3)), axis=1)**2)

if __name__ == '__main__':
    #unittest.main()

    
    #mu = np.linspace(-1, 1, 100)
    #plt.plot(mu, s_becke(mu), label='Becke')
    #plt.plot(mu, s_stratmann(mu), label='Stratmann')
    #plt.legend()
    #plt.show()

    R1 = np.array([0, 0, 0])
    R2 = np.array([0, 0, 1])
    a1 = 1.2
    a2 = 2.1

    #r = np.random.randn(3,10)
    #f = func(r, R1, R2, a1, a2)
    #print(f)

    r_rad, w_rad = baker(20, 7, 2)
    r_ang, w_ang = delley(5)

    r = []
    w = []
    for irad in range(len(r_rad)):
        for iang in range(len(r_ang)):
            r.append(r_rad[irad] * np.array(r_ang[iang]))
            w.append(w_rad[irad] * w_ang[iang])

    r = np.array(r)
    w = np.array(w)

    grid1 = r + R1
    grid2 = r + R2

    #val = 0.0
    #ref = (np.pi/a1)**1.5
    #for wi, ri in zip(w, grid1):
    #    val += wi * np.exp(-a1*np.linalg.norm(ri-R1)**2)

    #val *= 4 * np.pi
    #print(f'val = {val}')
    #print(f'ref = {ref}')


    val = 0.0
    for wi, ri in zip(w, grid1):
        w_becke = becke_part([R1, R2], ri)
        val += wi * func(ri, R1, R2, a1, a2)[0] * w_becke[0]

    for wi, ri in zip(w, grid2):
        w_becke = becke_part([R1, R2], ri)
        val += wi * func(ri, R1, R2, a1, a2)[0] * w_becke[1]

    val *= 4 * np.pi
    ref = (np.pi/a1)**1.5 + (np.pi/a2)**1.5
    print(f'val = {val}')
    print(f'ref = {ref}')




    #print(grid.shape)
    #print(weight.shape)

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(r[:,0], r[:,1], r[:,2])
    #plt.show()



