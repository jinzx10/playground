from radial import baker
from delley import delley

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
    s = s_becke
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

class TestBecke(unittest.TestCase):
    pass


if __name__ == '__main__':
    #unittest.main()

    
    mu = np.linspace(-1, 1, 100)
    plt.plot(mu, s_becke(mu), label='Becke')
    plt.plot(mu, s_stratmann(mu), label='Stratmann')
    plt.legend()
    plt.show()
