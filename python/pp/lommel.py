import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson

def king_smith_A(l, a, b, R):
    '''
                           / R
        A[a,b] = (a*b)^2 * |    dr jl(ar) * jl(br) * r^2
                           / 0
    
    '''
    if a == b:
        aR = a * R
        jl = spherical_jn(l, aR)
        jlp1 = spherical_jn(l+1, aR)
        return 0.5 * a * aR**2 * (
                aR*(jl**2 + jlp1**2) - (2*l+1) * jl * jlp1
                )

    aR = a*R
    bR = b*R
    return (a*b*R)**2 / (a**2-b**2) * (
            a * spherical_jn(l+1, aR) * spherical_jn(l, bR) -
            b * spherical_jn(l, aR) * spherical_jn(l+1, bR)
            )


############################################################
############################################################

import unittest

class _Test(unittest.TestCase):

    def test_king_smith_A(self):
        R = 2.77
        l = 1
        
        # integration grid for reference numerical calculation
        nr = 10000
        r = np.linspace(0, R, nr)
        
        ######################################################
        #           diagonal case (a == b)
        ######################################################
        a = 1.5
        f = spherical_jn(l, a*r) * spherical_jn(l, a*r) * r**2
        A = simpson(f, x=r) * a**4
        self.assertLess(np.abs(A - king_smith_A(l, a, a, R)), 1e-8)
        
        ######################################################
        #           non-diagonal case (a != b)
        ######################################################
        a = 3.5
        b = 2.3
        
        f = spherical_jn(l, a*r) * spherical_jn(l, b*r) * r**2
        A = simpson(f, x=r) * (a*b)**2
        self.assertLess(np.abs(A - king_smith_A(l, a, b, R)), 1e-8)


if __name__ == '__main__':
    unittest.main()

