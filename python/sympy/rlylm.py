import sympy as sp
import numpy as np
from scipy.special import sph_harm_y

def cart_to_angle(r):
    rnorm = np.linalg.norm(r)
    if rnorm < 1e-8:
        return 0.0, 0.0
    theta = np.arccos(r[2] / np.linalg.norm(r))
    phi = np.arctan2(r[1], r[0])
    return theta, phi


def ylmreal_scipy(l, m, theta, phi):
    if m == 0:
        return np.real(sph_harm_y(l, m, theta, phi))
    elif m > 0:
        return np.sqrt(2) * np.real(sph_harm_y(l, m, theta, phi))
    else:
        return np.sqrt(2) * np.imag(sph_harm_y(l, -m, theta, phi))


def rlylm_scipy(lmax, r):
    nr = np.size(r, 0)
    rlylm = np.zeros((nr, (lmax+1)**2))

    for ir in range(nr):
        theta, phi = cart_to_angle(r[ir])
        rmod = np.linalg.norm(r[ir])
        for l in range(lmax+1):
            for im in range(0, 2*l+1): # 0, 1, -1, ..., l, -l
                m = (im+1)//2 * (-1)**(im+1)
                ilm = l*l + im
                rlylm[ir, ilm] = rmod**l * ylmreal_scipy(l, m, theta, phi) * np.sqrt(4 * np.pi / (2*l+1))
    return rlylm


def rlylm_sympy(lmax, r):
    '''

    '''
    nr = np.size(r, 0)
    rmod = np.array([np.linalg.norm(r[ir,:]) for ir in range(nr)])
    rlylm = np.zeros((nr, (lmax+1)**2))

    # Define spherical harmonic functions
    l_, m_, theta_, phi_ = sp.symbols('l_ m_ theta_ phi_', real=True)
    
    # real spherical harmonics
    Z_lm = sp.Znm(l_, m_, theta_, phi_)

    col = 0
    for l in range(lmax+1):
        for im in range(0, 2*l+1): # 0, 1, -1, ..., l, -l
            m = (im+1)//2 * (-1)**(im+1)
            Z = Z_lm.subs({l_: l, m_: m})

            # a sign convention difference exists between sympy's
            # real spherical harmonics and ours
            sgn = -1 if (m < 0 and m%2==0) else 1
            for ir in range(nr):
                theta, phi = cart_to_angle(r[ir])
                Zval = sp.re(Z.subs({theta_: theta, phi_: phi}).evalf())
                rlylm[ir, col] = Zval * rmod[ir]**l * sgn * sp.sqrt(4 * sp.pi / (2*l+1))
            col += 1
    return rlylm


def rlylm_sym(lmax):
    '''
    Symbolic expression of real solid harmonics.

    '''
    x, y, z = sp.symbols('x y z')
    r2 = x*x + y*y + z*z
    rlylm = [None] * (lmax+1)**2
    rlylm[0] = 1
    rlylm[1] = z
    rlylm[2] = -x
    rlylm[3] = -y

    for l in range(2, lmax+1):
        idx_l = l*l
        idx_l_1 = (l - 1) * (l - 1)
        idx_l_2 = (l - 2) * (l - 2)

        # R(l,m) for m = 0, 1, -1, ..., l-2, 2-l
        fac1 = 2 * l - 1
        for im in range(0, 2 * l - 3):
            mabs = (im + 1) // 2  # 0, 1, 1, 2, 2, ...
            rlylm[idx_l+im] = (fac1 * z * rlylm[idx_l_1+im] - sp.sqrt((l-1-mabs)*(l-1+mabs)) * r2 * rlylm[idx_l_2+im]) / sp.sqrt((l + mabs) * (l - mabs))
            rlylm[idx_l+im] = rlylm[idx_l+im].simplify()

        # R(l,l-1) and R(l,1-l)
        fac = sp.sqrt(2*l-1)
        rlylm[idx_l+2*l-3] = (fac * z * rlylm[idx_l-2]).simplify()  
        rlylm[idx_l+2*l-2] = (fac * z * rlylm[idx_l-1]).simplify()

        # R(l,l) and R(l,-l)
        fac = -sp.sqrt(sp.Rational(2*l-1,2*l))
        rlylm[idx_l+2*l-1] = (fac * (x * rlylm[idx_l-2] - y * rlylm[idx_l-1])).simplify() 
        rlylm[idx_l+2*l  ] = (fac * (x * rlylm[idx_l-1] + y * rlylm[idx_l-2])).simplify()

    return rlylm


def rlylm_num(lmax, r):
    nr = np.size(r, 0)

    rlylm = np.zeros((nr, (lmax+1)**2))

    rlylm[:,0] = 1.0
    if lmax == 0:
        return rlylm

    rlylm[:,1] = r[:,2]
    rlylm[:,2] = -r[:,0]
    rlylm[:,3] = -r[:,1]
    if lmax == 1:
        return rlylm

    rmod2 = np.sum(r*r, axis=1)
    for l in range(2, lmax+1):
        idx_l = l*l
        idx_l_1 = (l - 1) * (l - 1)
        idx_l_2 = (l - 2) * (l - 2)

        # R(l,m) for m = 0, 1, -1, ..., l-2, 2-l
        fac1 = 2 * l - 1
        for im in range(0, 2 * l - 3):
            mabs = (im + 1) // 2  # 0, 1, 1, 2, 2, ...
            rlylm[:,idx_l+im] = (fac1 * r[:,2] * rlylm[:,idx_l_1+im] - np.sqrt((l-1-mabs)*(l-1+mabs)) * rmod2 * rlylm[:,idx_l_2+im]) \
                                / np.sqrt((l + mabs) * (l - mabs))

        # R(l,l-1) and R(l,1-l)
        fac = np.sqrt(2*l-1)
        rlylm[:,idx_l+2*l-3] = fac * r[:,2] * rlylm[:,idx_l-2]
        rlylm[:,idx_l+2*l-2] = fac * r[:,2] * rlylm[:,idx_l-1]

        # R(l,l) and R(l,-l)
        fac = -np.sqrt((2*l-1)/(2*l))
        rlylm[:,idx_l+2*l-1] = fac * (r[:,0] * rlylm[:,idx_l-2] - r[:,1] * rlylm[:, idx_l-1])
        rlylm[:,idx_l+2*l  ] = fac * (r[:,0] * rlylm[:,idx_l-1] + r[:,1] * rlylm[:, idx_l-2])

    return rlylm

lmax = 12
nr = 20

r = np.random.rand(nr, 3)

rlylm_ref = np.zeros((nr, (lmax+1)**2))
rlylm_expr = rlylm_sym(lmax)
x, y, z = sp.symbols('x y z')
for i in range((lmax+1)**2):
    f = sp.lambdify((x, y, z), rlylm_expr[i], "numpy")
    rlylm_ref[:,i] = f(r[:,0], r[:,1], r[:,2])

rlylm = rlylm_num(lmax, r)
abs_error = np.abs(rlylm - rlylm_ref)
max_abs_error = np.linalg.norm(abs_error, np.inf)
print(f'max abs error = {max_abs_error}')

rel_error = np.abs(rlylm - rlylm_ref) / np.maximum(np.abs(rlylm), np.abs(rlylm_ref))
max_rel_error = np.linalg.norm(rel_error, np.inf)
print(f'max rel error = {max_rel_error}')


rlylm_spnum = rlylm_sympy(lmax, r)
abs_error = np.abs(rlylm_spnum - rlylm_ref)
max_abs_error = np.linalg.norm(abs_error, np.inf)
print(f'max abs error = {max_abs_error}')


rlylm_sci = rlylm_scipy(lmax, r)
abs_error = np.abs(rlylm_sci - rlylm_ref)
max_abs_error = np.linalg.norm(abs_error, np.inf)
print(f'max abs error = {max_abs_error}')
