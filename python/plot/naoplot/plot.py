import numpy as np
from scipy.special import sph_harm
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from orbio import read_nao


def real_sph_harm(l, m, polar, azimuth):
    if m > 0:
        return np.sqrt(2) * np.real(sph_harm(m, l, azimuth, polar))
    elif m < 0:
        return np.sqrt(2) * np.imag(sph_harm(m, l, azimuth, polar))
    else:
        return np.real(sph_harm(m, l, azimuth, polar))


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    polar = np.arccos(z/r)
    azimuth = np.arctan2(y, x)
    return r, polar, azimuth


####################################################
#               read orbital file
####################################################
#nao = read_nao('./O_gga_10au_100Ry_2s2p1d.orb')
#nao = read_nao('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/26_Fe_DZP/Fe_gga_10au_100Ry_4s2p2d1f.orb')
nao = read_nao('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/72_Hf_DZP/Hf_gga_10au_100Ry_4s2p2d2f1g.orb')
r = nao['dr'] * np.arange(nao['nr'])
chi = nao['chi']

#plt.plot(r, chi[4][0])
#plt.show()
#
#exit()


####################################################
#                   3d mesh grid
####################################################
w = 5
ngrid = 40
x, y, z = np.mgrid[-w:w:ngrid*1j, -w:w:ngrid*1j, -w:w:ngrid*1j]

x = x.flatten()
y = y.flatten()
z = z.flatten()

####################################################
#                   orbital
####################################################
l = 4
zeta = 0
m = 0

# orbital value on the grid
value = np.zeros_like(x)

chi_spline = CubicSpline(r, chi[l][zeta])
for i in range(ngrid**3):
    r, polar, azimuth = cart2sph(x[i], y[i], z[i])
    if r < nao['rcut']:
        value[i] = chi_spline(r) * real_sph_harm(l, m, polar, azimuth)

####################################################
#                   plot
####################################################
fig = go.Figure(data=go.Volume(
    x=x,
    y=y,
    z=z,
    value=value,
    isomin=-0.1,
    isomax=0.1,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=30, # needs to be a large number for good volume rendering
    ))
fig.show()


############################################################
#                           Test
############################################################
import unittest

class _TestPlot(unittest.TestCase):

    def test_real_sph_harm(self):
        n = 100

        x = np.random.randn(n)
        y = np.random.randn(n)
        z = np.random.randn(n)
        r = np.sqrt(x**2 + y**2 + z**2)

        polar = np.arccos(z/r)
        azimuth = np.arctan2(y, x)

        # check:
        # Y_{1,1}  = -sqrt(3/4/pi) * x/r
        # Y_{1,0}  =  sqrt(3/4/pi) * z/r
        # Y_{1,-1} = -sqrt(3/4/pi) * y/r
        for i in range(n):
            self.assertAlmostEqual(real_sph_harm(1, 1, polar[i], azimuth[i]), \
                -np.sqrt(3/(4*np.pi)) * x[i]/r[i])
            self.assertAlmostEqual(real_sph_harm(1, 0, polar[i], azimuth[i]), \
                np.sqrt(3/(4*np.pi)) * z[i]/r[i])
            self.assertAlmostEqual(real_sph_harm(1, -1, polar[i], azimuth[i]), \
                -np.sqrt(3/(4*np.pi)) * y[i]/r[i])


if __name__ == '__main__':
    unittest.main()

