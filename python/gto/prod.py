import numpy as np
from math import comb
from scipy.special import sph_harm_y

def gauss_prod(alpha, A, beta, B):
    r'''
    Gaussian product rule.

    This function returns the prefactor "K" and new center "C"
    of the product of two Gaussians:

    exp[-alpha*(r-A)^2] * exp[-beta*(r-B)^2]
        = K * exp[-(alpha+beta)*(r-C)^2]

    '''
    gamma = alpha * beta / (alpha + beta)
    rAB = np.linalg.norm(A-B)
    K = np.exp(-gamma*rAB**2)
    C = (alpha * A + beta * B) / (alpha + beta)
    return K, C


def S_prod_expand(alpha, A, l1, m1, beta, B, l2, m2):
    r'''
    Expansion of a product of two real solid harmonics.

    The product of two real solid harmonics has a finite expansion

    '''





def sGTO_prod_expand(alpha, A, l1, m1, beta, B, l2, m2):
    r'''
    Expansion of a product of two spherical GTOs.

    The product of two real solid harmonics has a finite expansion

    '''
    K, C = gauss_prod(alpha, A, beta, B)
