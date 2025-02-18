import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson


def lommel_j(l, a, b, R):
    r'''
    Compute the following Lommel integral

            / R
            |    dr jl(ar) * jl(br) * r^2
            / 0

    '''
    # FIXME
    #
    # 1. One may use a recursive formula to combine the computation
    # of j_{l+1} and j_{l}, which saves some time.
    #
    # 2. The code may have significant numerical error when `a` is
    # very close (but not equal) to `b`. The special case a == b is
    # a 0/0 limit of the general expression. As a general subroutine
    # for computing such integral, it might be necessary to have a
    # branched algorithm based on |a-b| and some finite threshold,
    # instead of the if-else below (though this is probably adequate
    # for use in the King-Smith method).
    if a == b:
        aR = a * R
        jl = spherical_jn(l, aR)
        jlp1 = spherical_jn(l+1, aR)
        return 0.5 * R**2 / a * (
                aR*(jl**2 + jlp1**2) - (2*l+1) * jl * jlp1
                )

    aR = a*R
    bR = b*R
    return R**2 / (a**2-b**2) * (
            a * spherical_jn(l+1, aR) * spherical_jn(l, bR) -
            b * spherical_jn(l, aR) * spherical_jn(l+1, bR)
            )


def lommel_j_alt(l, a, b, R):
    '''
    Same as lommel_j, but using an equivalent expression

    '''
    # FIXME
    # Similar implementation advice applies. The evaluation of
    # the derivative of the spherical Bessel function actually
    # uses the same recursive formula.
    if a == b:
        aR = a * R
        jl = spherical_jn(l, aR)
        jlp = spherical_jn(l, aR, True)
        return 0.5 * R**2 / a * (
                jl**2*(aR-l*(l+1)/aR) + jlp**2 * aR + jlp * jl
                )

    aR = a*R
    bR = b*R
    return R**2 / (a**2-b**2) * (
            b * spherical_jn(l, aR) * spherical_jn(l, bR, True) - 
            a * spherical_jn(l, aR, True) * spherical_jn(l, bR)
            )


############################################################
############################################################

import unittest

class _Test(unittest.TestCase):

    def test_lommel_j(self):
        R = 2.77
        l = 1
        
        # integration grid for reference numerical calculation
        nr = 10000
        r = np.linspace(0, R, nr)
        tol = 1e-8
        
        ######################################################
        #           diagonal case (a == b)
        ######################################################
        a = 1.5
        f = spherical_jn(l, a*r) * spherical_jn(l, a*r) * r**2
        A = simpson(f, x=r)
        self.assertLess(np.abs(A - lommel_j(l, a, a, R)), tol)
        
        ######################################################
        #           non-diagonal case (a != b)
        ######################################################
        a = 3.5
        b = 2.3
        
        f = spherical_jn(l, a*r) * spherical_jn(l, b*r) * r**2
        A = simpson(f, x=r)
        self.assertLess(np.abs(A - lommel_j(l, a, b, R)), tol)


    def test_lommel_j_alt(self):
        R = 2.77
        l = 1
        
        # integration grid for reference numerical calculation
        nr = 10000
        r = np.linspace(0, R, nr)
        tol = 1e-8
        
        ######################################################
        #           diagonal case (a == b)
        ######################################################
        a = 1.5
        f = spherical_jn(l, a*r) * spherical_jn(l, a*r) * r**2
        A = simpson(f, x=r)
        self.assertLess(np.abs(A - lommel_j_alt(l, a, a, R)), tol)
        
        ######################################################
        #           non-diagonal case (a != b)
        ######################################################
        a = 3.5
        b = 2.3
        
        f = spherical_jn(l, a*r) * spherical_jn(l, b*r) * r**2
        A = simpson(f, x=r)
        self.assertLess(np.abs(A - lommel_j_alt(l, a, b, R)), tol)

if __name__ == '__main__':
    unittest.main()

