import sympy as sp
import numpy as np
from scipy.special import sph_harm_y

def cart_to_angle(r):
    rnorm = np.linalg.norm(r)
    if rnorm < 1e-9:
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
    '''
    Real solid harmonics based on scipy's sph_harm_y.

    '''
    nr = np.size(r, 0)
    rlylm = np.zeros((nr, (lmax+1)**2))

    for ir in range(nr):
        theta, phi = cart_to_angle(r[ir])
        rmod = np.linalg.norm(r[ir])
        for l in range(lmax+1):
            for im in range(0, 2*l+1): # 0, 1, -1, ..., l, -l
                m = (im+1)//2 * (-1)**(im+1)
                ilm = l*l + im
                rlylm[ir, ilm] = rmod**l * ylmreal_scipy(l, m, theta, phi) \
                                * np.sqrt(4 * np.pi / (2*l+1))
    return rlylm


def rlylm_sym(lmax):
    '''
    Symbolic expression of real solid harmonics.

    '''
    x, y, z = sp.symbols('x y z')
    r2 = x*x + y*y + z*z
    rlylm = sp.zeros(1,(lmax+1)**2)
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

    r2 = np.sum(r*r, axis=1)
    for l in range(2, lmax+1):
        idx_l = l*l
        idx_l_1 = (l - 1) * (l - 1)
        idx_l_2 = (l - 2) * (l - 2)

        # R(l,m) for m = 0, 1, -1, ..., l-2, 2-l
        fac1 = 2 * l - 1
        for im in range(0, 2 * l - 3):
            mabs = (im + 1) // 2  # 0, 1, 1, 2, 2, ...
            rlylm[:,idx_l+im] = (fac1 * r[:,2] * rlylm[:,idx_l_1+im] - np.sqrt((l-1-mabs)*(l-1+mabs)) * r2 * rlylm[:,idx_l_2+im]) \
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


def grad_rlylm_num(lmax, r):
    '''
    Real solid harmonics and their Cartesian gradients.

    '''
    nr = np.size(r, 0)

    rlylm = np.zeros((nr, (lmax+1)**2))
    grad_rlylm = np.zeros((3, nr, (lmax+1)**2))

    rlylm[:,0] = 1.0
    if lmax == 0:
        return rlylm, grad_rlylm

    rlylm[:,1] = r[:,2]
    rlylm[:,2] = -r[:,0]
    rlylm[:,3] = -r[:,1]
    grad_rlylm[2,:,1] = 1.0
    grad_rlylm[0,:,2] = -1.0
    grad_rlylm[1,:,3] = -1.0
    if lmax == 1:
        return rlylm, grad_rlylm

    r2 = np.sum(r*r, axis=1)
    for l in range(2, lmax+1):
        idx_l = l*l
        idx_l_1 = (l - 1) * (l - 1)
        idx_l_2 = (l - 2) * (l - 2)

        # R(l,m) for m = 0, 1, -1, ..., l-2, 2-l
        fac1 = 2 * l - 1
        for im in range(0, 2 * l - 3):
            mabs = (im + 1) // 2  # 0, 1, 1, 2, 2, ...
            fac2 = np.sqrt((l-1-mabs)*(l-1+mabs))
            fac3 = 1.0 / np.sqrt((l + mabs) * (l - mabs))
            rlylm[:,idx_l+im] = (fac1 * r[:,2] * rlylm[:,idx_l_1+im] \
                                - fac2 * r2 * rlylm[:,idx_l_2+im]) * fac3
            grad_rlylm[0,:,idx_l+im] = fac3 * \
                (fac1 * r[:,2] * grad_rlylm[0,:,idx_l_1+im] \
                - fac2 * (2*r[:,0]*rlylm[:,idx_l_2+im] + r2*grad_rlylm[0,:,idx_l_2+im]))
            grad_rlylm[1,:,idx_l+im] = fac3 * \
                (fac1 * r[:,2] * grad_rlylm[1,:,idx_l_1+im] \
                - fac2 * (2*r[:,1]*rlylm[:,idx_l_2+im] + r2*grad_rlylm[1,:,idx_l_2+im]))
            grad_rlylm[2,:,idx_l+im] = fac3 * \
                (fac1 * (rlylm[:,idx_l_1+im] + r[:,2] * grad_rlylm[2,:,idx_l_1+im]) \
                - fac2 * (2*r[:,2]*rlylm[:,idx_l_2+im] + r2*grad_rlylm[2,:,idx_l_2+im]))

        # R(l,l-1) and R(l,1-l)
        fac = np.sqrt(2*l-1)
        rlylm[:,idx_l+2*l-3] = fac * r[:,2] * rlylm[:,idx_l-2]
        rlylm[:,idx_l+2*l-2] = fac * r[:,2] * rlylm[:,idx_l-1]
        grad_rlylm[0,:,idx_l+2*l-3] = fac * r[:,2] * grad_rlylm[0,:,idx_l-2]
        grad_rlylm[1,:,idx_l+2*l-3] = fac * r[:,2] * grad_rlylm[1,:,idx_l-2]
        grad_rlylm[2,:,idx_l+2*l-3] = fac * (rlylm[:,idx_l-2] \
                                            + r[:,2] * grad_rlylm[2,:,idx_l-2])
        grad_rlylm[0,:,idx_l+2*l-2] = fac * r[:,2] * grad_rlylm[0,:,idx_l-1]
        grad_rlylm[1,:,idx_l+2*l-2] = fac * r[:,2] * grad_rlylm[1,:,idx_l-1]
        grad_rlylm[2,:,idx_l+2*l-2] = fac * (rlylm[:,idx_l-1] \
                                            + r[:,2] * grad_rlylm[2,:,idx_l-1])

        # R(l,l) and R(l,-l)
        fac = -np.sqrt((2*l-1)/(2*l))
        rlylm[:,idx_l+2*l-1] = fac * (r[:,0] * rlylm[:,idx_l-2] \
                                    - r[:,1] * rlylm[:, idx_l-1])
        rlylm[:,idx_l+2*l  ] = fac * (r[:,0] * rlylm[:,idx_l-1] \
                                    + r[:,1] * rlylm[:, idx_l-2])
        grad_rlylm[0,:,idx_l+2*l-1] = fac * (rlylm[:,idx_l-2] \
                + r[:,0] * grad_rlylm[0,:,idx_l-2] \
                - r[:,1] * grad_rlylm[0,:,idx_l-1])
        grad_rlylm[1,:,idx_l+2*l-1] = fac * (-rlylm[:,idx_l-1] \
                + r[:,0] * grad_rlylm[1,:,idx_l-2] \
                - r[:,1] * grad_rlylm[1,:,idx_l-1])
        grad_rlylm[2,:,idx_l+2*l-1] = fac * (
                + r[:,0] * grad_rlylm[2,:,idx_l-2] \
                - r[:,1] * grad_rlylm[2,:,idx_l-1])
        grad_rlylm[0,:,idx_l+2*l] = fac * (rlylm[:,idx_l-1] \
                + r[:,0] * grad_rlylm[0,:,idx_l-1] \
                + r[:,1] * grad_rlylm[0,:,idx_l-2])
        grad_rlylm[1,:,idx_l+2*l] = fac * (rlylm[:,idx_l-2] \
                + r[:,0] * grad_rlylm[1,:,idx_l-1] \
                + r[:,1] * grad_rlylm[1,:,idx_l-2])
        grad_rlylm[2,:,idx_l+2*l] = fac * (
                + r[:,0] * grad_rlylm[2,:,idx_l-1] \
                + r[:,1] * grad_rlylm[2,:,idx_l-2])

    return rlylm, grad_rlylm


def random_rational(denom, num_min, num_max):
    num = np.random.randint(num_min, num_max+1)
    return sp.Rational(num, denom)


lmax = 12
nr = 20

# generate r as random rational
denom   =  100000000
num_max =  300000000
num_min = -300000000
r_sym = sp.Matrix(nr, 3, lambda i, j: random_rational(denom, num_min, num_max))

x, y, z = sp.symbols('x y z')
rlylm_expr = rlylm_sym(lmax)
rlylm_ref_sym = sp.zeros(nr, (lmax+1)**2)

for ir in range(nr):
    rlylm_ref_sym[ir,:] = rlylm_expr.subs(\
            {x: r_sym[ir,0], y: r_sym[ir,1], z: r_sym[ir,2]})

# reference numerical values
rlylm_ref = np.array(rlylm_ref_sym.evalf(), dtype=float)
#print(rlylm_expr)
#print(rlylm_ref_sym)
#print(rlylm_ref)


r = np.array(r_sym.evalf(), dtype=float)
#print(r)

# recurrence formula
rlylm = rlylm_num(lmax, r)
abs_error = np.abs(rlylm - rlylm_ref)
max_abs_error = np.linalg.norm(abs_error, np.inf)
print(f'recur max abs error = {max_abs_error}')

rel_error = abs_error / np.maximum(np.abs(rlylm), np.abs(rlylm_ref))
max_rel_error = np.linalg.norm(rel_error, np.inf)
print(f'recur max rel error = {max_rel_error}')


# based on scipy's implementation of spherical harmonics
rlylm_sci = rlylm_scipy(lmax, r)
abs_error = np.abs(rlylm_sci - rlylm_ref)
max_abs_error = np.linalg.norm(abs_error, np.inf)
print(f'scipy max abs error = {max_abs_error}')

rel_error = abs_error / np.maximum(np.abs(rlylm_sci), np.abs(rlylm_ref))
max_rel_error = np.linalg.norm(rel_error, np.inf)
print(f'scipy max rel error = {max_rel_error}')

################################################################
#           gradients of real solid harmonics
################################################################
grad_rlylm_x_expr = rlylm_expr.diff(x)
grad_rlylm_y_expr = rlylm_expr.diff(y)
grad_rlylm_z_expr = rlylm_expr.diff(z)
#print(grad_rlylm_x_expr)

grad_rlylm_x_ref_sym = sp.zeros(nr, (lmax+1)**2)
grad_rlylm_y_ref_sym = sp.zeros(nr, (lmax+1)**2)
grad_rlylm_z_ref_sym = sp.zeros(nr, (lmax+1)**2)
for ir in range(nr):
    grad_rlylm_x_ref_sym[ir,:] = grad_rlylm_x_expr.subs(\
            {x: r_sym[ir,0], y: r_sym[ir,1], z: r_sym[ir,2]})
    grad_rlylm_y_ref_sym[ir,:] = grad_rlylm_y_expr.subs(\
            {x: r_sym[ir,0], y: r_sym[ir,1], z: r_sym[ir,2]})
    grad_rlylm_z_ref_sym[ir,:] = grad_rlylm_z_expr.subs(\
            {x: r_sym[ir,0], y: r_sym[ir,1], z: r_sym[ir,2]})

grad_rlylm_ref = np.zeros((3, nr, (lmax+1)**2))
grad_rlylm_ref[0,:,:] = np.array(grad_rlylm_x_ref_sym.evalf(), dtype=float)
grad_rlylm_ref[1,:,:] = np.array(grad_rlylm_y_ref_sym.evalf(), dtype=float)
grad_rlylm_ref[2,:,:] = np.array(grad_rlylm_z_ref_sym.evalf(), dtype=float)
#print(grad_rlylm_ref)


_, grad_rlylm = grad_rlylm_num(lmax, r)
abs_error = np.abs(grad_rlylm - grad_rlylm_ref)
max_abs_error = np.linalg.norm(abs_error.reshape(-1), np.inf)
print(f'grad max abs error = {max_abs_error}')

rel_error = abs_error / (np.maximum(np.abs(grad_rlylm), np.abs(grad_rlylm_ref)) + 1e-16)
max_rel_error = np.linalg.norm(rel_error.reshape(-1), np.inf)
print(f'grad max rel error = {max_rel_error}')

