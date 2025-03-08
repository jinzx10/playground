import numpy as np
from scipy.special import sph_harm_y

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


def r2s(m, mp):
    # spherical harmonics: real-to-standard transformation
    return int((m == 0) * (mp == 0)) + 1/np.sqrt(2) * (
            int(m > 0) * (-1)**m * (int(m == mp) + 1j * int(m == -mp)) + 
            int(m < 0) * (int(m == -mp) - 1j * int(m == mp))
            )


def s2r(m, mp):
    # standard-to-real
    return np.conj(r2s(mp, m))

