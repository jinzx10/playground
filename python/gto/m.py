import numpy as np
from math import comb

def r2s(m, mp):
    # spherical harmonics: real-to-standard transformation
    return int((m == 0) * (mp == 0)) + 1/np.sqrt(2) * (
            int(m > 0) * (-1)**m * (int(m == mp) + 1j * int(m == -mp)) + 
            int(m < 0) * (int(m == -mp) - 1j * int(m == mp))
            )


def s2r(m, mp):
    # standard-to-real
    return np.conj(r2s(mp, m))



lmax = 1

for l in range(lmax+1):
    for lp in range(lmax+1):
        for mu in range(-l, l+1):
            for nu in range(-lp, lp+1):
                for lam in range(lp-l, l-lp+1):
                    val = 0
                    for m in range(-l, l+1):
                        for mp in range(-lp, lp+1):
                            val += s2r(mu, m) * r2s(mp, nu) * r2s(m-mp, lam) * np.sqrt(comb(l+m, lp+mp) * comb(l-m, lp-mp))

                    #if val == 0:
                    if True:
                        print(f'l={l}  lp={lp}  mu={mu:2}  nu={nu:2}  lam={lam:2}  M^2={np.real(val)**2: 20.15f}')

