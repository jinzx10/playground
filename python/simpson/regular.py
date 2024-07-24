import numpy as np
from scipy.integrate import simpson

# regularly-spaced
def simp(f, dx):

    n = len(f)
    if n == 2:
        return 0.5 * dx * (f[0] + f[1])

    assert(n > 2)
    if n % 2 == 1:
        # odd number of points (even number of intervals)
        # composite Simpson's rule
        return dx / 3.0 * (f[0] + 4.0*np.sum(f[1:n-1:2]) + 2.0*np.sum(f[2:n-2:2]) + f[n-1])
    else:
        # n is even and n >= 4
        # composite Simpson's rule for the first n-1 points plus the contribution from the last interval
        return simp(f[:-1], dx) + dx * (5.0 / 12.0 * f[-1] + 2.0 / 3.0 * f[-2] - f[-3] / 12.0)

def simpmix(f, dx):

    n = len(f)
    if n == 4:
        return 3.0 * dx / 8 * (f[0] + 3.0 * f[1] + 3.0 * f[2] + f[3])
    if n == 2:
        return 0.5 * dx * (f[0] + f[1])

    assert(n > 2)
    if n % 2 == 1:
        # odd number of points (even number of intervals)
        # composite Simpson's rule
        return dx / 3.0 * (f[0] + 4.0*np.sum(f[1:n-1:2]) + 2.0*np.sum(f[2:n-2:2]) + f[n-1])
    else:
        # n is even and n >= 6
        # composite Simpson's 1/3 rule for the first n-4 intervals plus Simpson's 3/8 rule for the last 3 intervals
        return simpmix(f[:-3], dx) + simpmix(f[-4:], dx)

def bench(ngrid, ntimes):
    err_scipy = np.zeros(ntimes)
    err_isimp = np.zeros(ntimes)
    err_msimp = np.zeros(ntimes)

    for i in range(ntimes):
        lower_limit = (np.random.randn() - 0.5) * np.pi
        upper_limit = lower_limit + 2*np.pi * np.random.rand()
        
        x = np.linspace(lower_limit, upper_limit, ngrid)
        y = np.sin(x)
        val_ref = np.cos(lower_limit) - np.cos(upper_limit)

        val_scipy = simpson(y, dx=x[1]-x[0])
        val_isimp = simp(y, x[1]-x[0])
        val_msimp = simpmix(y, x[1]-x[0])

        err_scipy[i] = abs(val_scipy - val_ref)
        err_isimp[i] = abs(val_isimp - val_ref)
        err_msimp[i] = abs(val_msimp - val_ref)

    print('scipy/isimp/simpmix= %i/%i/%i'%( np.sum( (err_scipy < err_isimp) & (err_scipy < err_msimp) ),\
                                            np.sum( (err_isimp < err_scipy) & (err_isimp < err_msimp) ),\
                                            np.sum( (err_msimp < err_scipy) & (err_msimp < err_isimp) )))

    print('errmax_scipy = %6.4e      errstd_scipy = %6.4e'%(np.max(err_scipy), np.std(err_scipy))) 
    print('errmax_isimp = %6.4e      errstd_isimp = %6.4e'%(np.max(err_isimp), np.std(err_isimp)))
    print('errmax_msimp = %6.4e      errstd_msimp = %6.4e'%(np.max(err_msimp), np.std(err_msimp)))


bench(100,10000)
bench(81,10000)


