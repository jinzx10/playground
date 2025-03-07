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
        for mp in range(-lp, lp+1):
            val += s2r(mu, m) * r2s(mp, nu) * r2s(m-mp, lam) * np.sqrt(comb(l+m, lp+mp) * comb(l-m, lp-mp))

    return val


def real_sph_harm(l, m, theta, phi):
    if m == 0:
        return sph_harm_y(l, 0, theta, phi)
    elif m > 0:
        return (-1)**m * np.real(sph_harm_y(l, m, theta, phi))
    else:
        return (-1)**m * np.imag(sph_harm_y(l, -m, theta, phi))


def solid_sph_harm(l, m, r):
    rabs = np.linalg.norm(r)
    theta = np.arccos(r[2]/rabs)
    phi = np.arctan2(r[1], r[0])
    return np.sqrt(4*np.pi/(2*l+1)) * rabs**l \
            * real_sph_harm(l, m, theta, phi)

#lmax = 1
#
#for l in range(lmax+1):
#    for lp in range(lmax+1):
#        for mu in range(-l, l+1):
#            for nu in range(-lp, lp+1):
#                for lam in range(lp-l, l-lp+1):
#                    val = 0
#                    for m in range(-l, l+1):
#                        for mp in range(-lp, lp+1):
#                            val += s2r(mu, m) * r2s(mp, nu) * r2s(m-mp, lam) * np.sqrt(comb(l+m, lp+mp) * comb(l-m, lp-mp))
#
#                    #if val == 0:
#                    if True:
#                        print(f'l={l}  lp={lp}  mu={mu:2}  nu={nu:2}  lam={lam:2}  M^2={np.real(val)**2: 20.15f}')
#

r1 = np.random.randn(3)
r2 = np.random.randn(3)

l = 3
m = -2 

ref = solid_sph_harm(l, m, r1+r2)

val = 0
for lp in range(l+1):
    for nu in range(-lp, lp+1):
        for lam in range(lp-l, l-lp+1):
            val += M(l, lp, m, nu, lam) * solid_sph_harm(lp, nu, r1) * solid_sph_harm(l-lp, lam, r2)

print(f'ref={ref: 20.15f}  val={val: 20.15f}  diff={ref-val: 20.15f}')






