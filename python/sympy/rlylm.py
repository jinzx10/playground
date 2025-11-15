import numpy as np
import sympy as sp

def cart_to_angle(r):
    rnorm = np.linalg.norm(r)
    if rnorm < 1e-8:
        return 0.0, 0.0
    theta = np.arccos(r[2] / np.linalg.norm(r))
    phi = np.arctan2(r[1], r[0])
    return theta, phi

def rlylm_ref(lmax, r, rmod):
    '''

    '''
    nr = len(rmod)
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
                rlylm[ir, col] = Zval * rmod[ir]**l * sgn
            col += 1
    return rlylm


def rlylm_real_batch(lmax, r, rmod):
    '''
    Adapted from ABACUS's implementation (ylm.cpp)

    '''

    nr = len(rmod)
    rlylm = np.zeros((nr, (lmax+1)**2))

    # R(0, 0) = sqrt(1/4pi)
    rlylm[:, 0] = np.sqrt(0.25 / np.pi)
    if lmax == 0:
        return rlylm

    # l=1, m: (0, 1, -1) -> (z, -x, -y) * sqrt(3/4pi)
    fac1 = np.sqrt(0.75 / np.pi)
    rlylm[:, 1] = fac1 * r[:, 2]
    rlylm[:, 2] = -fac1 * r[:, 0]
    rlylm[:, 3] = -fac1 * r[:, 1]
    if lmax == 1:
        return rlylm

    rmod2 = rmod * rmod # r^2
    for l in range(2, lmax + 1):
        # starting column index of l, l-1 and l-2
        idx_l = l * l
        idx_l_1 = (l - 1) * (l - 1)
        idx_l_2 = (l - 2) * (l - 2)

        # R(l,m) for m = 0, 1, -1, ..., l-2, 2-l
        fac1 = np.sqrt((2 * l - 1) * (2 * l + 1))
        fac2 = np.sqrt((2 * l - 3) * (2 * l - 1))
        for im in range(0, 2 * l - 3):
            mabs = (im + 1) // 2  # 0, 1, 1, 2, 2, ...
            rlylm[:,idx_l+im] = (r[:,2] * rlylm[:,idx_l_1+im] - np.sqrt((l-1-mabs)*(l-1+mabs))/fac2 * rmod2 * rlylm[:,idx_l_2+im]) * fac1/np.sqrt((l+mabs)*(l-mabs));

        # R(l,l-1) and R(l,1-l)
        fac = np.sqrt(2*l+1)
        rlylm[:,idx_l+2*l-3] = fac * r[:,2] * rlylm[:,idx_l-2]
        rlylm[:,idx_l+2*l-2] = fac * r[:,2] * rlylm[:,idx_l-1]

        # R(l,l) and R(l,-l)
        fac = -np.sqrt((2*l+1)/(2*l))
        rlylm[:,idx_l+2*l-1] = fac * (r[:,0] * rlylm[:,idx_l-2] - r[:,1] * rlylm[:, idx_l-1])
        rlylm[:,idx_l+2*l  ] = fac * (r[:,0] * rlylm[:,idx_l-1] + r[:,1] * rlylm[:, idx_l-2])

    return rlylm

lmax = 8
nr = 10
r = np.random.randn(nr, 3)
#r = np.array([[1,2,3]])
rmod = np.array([np.linalg.norm(r[ir,:]) for ir in range(nr)])


rlylm = rlylm_real_batch(lmax, r, rmod)
rlylm_ref = rlylm_ref(lmax, r, rmod)
#print(rlylm)
#print(rlylm_ref)

abs_error = np.abs(rlylm - rlylm_ref)
max_abs_error = np.linalg.norm(abs_error, np.inf)
print(f'max abs error = {max_abs_error}')

rel_error = np.abs(rlylm - rlylm_ref) / np.maximum(np.abs(rlylm), np.abs(rlylm_ref))
max_rel_error = np.linalg.norm(rel_error, np.inf)
print(f'max rel error = {max_rel_error}')


