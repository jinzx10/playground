import numpy as np
from math import comb
from scipy.special import sph_harm_y

def r2s(m, mp):
    # spherical harmonics: real-to-standard transformation
    return int((m == 0) * (mp == 0)) + 1/np.sqrt(2) * (
            int(m > 0) * (-1)**m * (int(m == mp) + 1j * int(m == -mp)) + 
            int(m < 0) * (int(m == -mp) - 1j * int(m == mp))
            )


def s2r(m, mp):
    # standard-to-real
    return np.conj(r2s(mp, m))


def M(l, lp, mu, nu, lam):
    val = 0
    for m in range(-l, l+1):
        for mp in range(max(-lp, m+lp-l), min(lp,m+l-lp)+1):
            val += s2r(mu, m) * r2s(mp, nu) * r2s(m-mp, lam) * np.sqrt(comb(l+m, lp+mp) * comb(l-m, lp-mp))

    return np.real(val)


def real_sph_harm(l, m, theta, phi):
    if m == 0:
        return np.real(sph_harm_y(l, 0, theta, phi))
    elif m > 0:
        return (-1)**m * np.sqrt(2) * np.real(sph_harm_y(l, m, theta, phi))
    else:
        return (-1)**m * np.sqrt(2) * np.imag(sph_harm_y(l, -m, theta, phi))


def solid_sph_harm(l, m, r):
    rabs = np.linalg.norm(r)
    theta = np.arccos(r[2]/rabs)
    phi = np.arctan2(r[1], r[0])
    return np.sqrt(4*np.pi/(2*l+1)) * rabs**l \
            * real_sph_harm(l, m, theta, phi)


###########################################################################

import unittest

class TestAddition(unittest.TestCase):

    def test_(self):

        l = 4
        m = -3

        r1 = np.random.randn(3)
        r2 = np.random.randn(3)
        ref = solid_sph_harm(l, m, r1+r2)
        
        val = 0.0
        for lp in range(l+1):
            for nu in range(-lp, lp+1):
                for lam in range(lp-l, l-lp+1):
                    MM = M(l, lp, m, nu, lam)
                    print(f'l={l}  m={m:2}  lp={lp}  nu={nu:2}  lam={lam:2}  M^2={MM**2:8.3f}')
                    val += M(l, lp, m, nu, lam) * solid_sph_harm(lp, nu, r1) * solid_sph_harm(l-lp, lam, r2)
        
        #print(f'ref={ref: 20.15f}  val={val: 20.15f}  diff={abs(ref-val): 20.15f}')

    def _test_all(self):

        lmax = 4
        n = 5

        for l in range(lmax+1):
            for m in range(-l, l+1):
                for i in range(n):
                    r1 = np.random.randn(3)
                    r2 = np.random.randn(3)
                    ref = solid_sph_harm(l, m, r1+r2)
        
                    val = 0.0
                    for lp in range(l+1):
                        for nu in range(-lp, lp+1):
                            for lam in range(lp-l, l-lp+1):
                                val += M(l, lp, m, nu, lam) * solid_sph_harm(lp, nu, r1) * solid_sph_harm(l-lp, lam, r2)
                    
                    #print(f'ref={ref: 20.15f}  val={val: 20.15f}  diff={abs(ref-val): 20.15f}')
                    self.assertAlmostEqual(ref, val, 12)



if __name__ == '__main__':
    unittest.main()


