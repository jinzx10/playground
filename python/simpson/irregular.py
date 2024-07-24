import numpy as np
from scipy.integrate import simpson

# irregularly-spaced
# Cartwright's correction for even number of points, see wikipedia
def simp2(f, x):

    n = len(x)
    if n == 2:
        return 0.5 * (x[1] - x[0]) * (f[0] + f[1])

    assert(n >= 2)
    if n % 2 == 1:
        h = np.diff(x)
        hodd = h[1::2]
        heven = h[0::2]
        hoe = hodd / heven
        heo = heven / hodd
        return np.sum( (heven+hodd)/6*( (2-hoe)*f[0:-1:2] + (2+hoe+heo)*f[1:-1:2] + (2-heo)*f[2::2]) )
    else:
        h1 = x[-1] - x[-2]
        h2 = x[-2] - x[-3]
        return simp2(f[:-1], x[:-1]) \
                + f[-1] * ( 2.0*h1**2 + 3.0*h1*h2 ) / ( 6.0 * (h1 + h2) ) \
                + f[-2] * ( h1**2 + 3.0*h1*h2 ) / ( 6.0 * h2 )\
                - f[-3] * ( h1**3 ) / ( 6.0 * h2 * (h1 + h2) )
# Simpson 1/3 + 3/8
def simpmix2(f, h):

    n = len(f)
    assert(n >= 2)

    if n == 4:
        w = h[0] + h[1] + h[2]
        return w / 12.0 * ( 2.0 + ((h[1]+h[2])/h[0]-1.0) * (h[2]/(h[0]+h[1])-1.0) ) * f[0] + \
                w**3 / 12.0 * (h[0]+h[1]-h[2]) / (h[0]*h[1]*(h[1]+h[2])) * f[1] + \
                w**3 / 12.0 * (h[2]+h[1]-h[0]) / (h[2]*h[1]*(h[1]+h[0])) * f[2] + \
                w / 12.0 * ( 2.0 + ((h[1]+h[0])/h[2]-1.0) * (h[0]/(h[2]+h[1])-1.0) ) * f[3];

    if n == 2:
        return 0.5 * h[0] * (f[0] + f[1])

    if n % 2 == 1:
        hodd = h[1::2]
        heven = h[0::2]
        hoe = hodd / heven
        heo = heven / hodd
        return np.sum( (heven+hodd)/6*( (2-hoe)*f[0:-1:2] + (2+hoe+heo)*f[1:-1:2] + (2-heo)*f[2::2]) )
    else:
        return simpmix2(f[:-3], h[:-3]) + simpmix2(f[-4:], h[-3:])

def bench2(ngrid, ntimes):
    err_scipy = np.zeros(ntimes)
    err_isimp = np.zeros(ntimes)
    err_msimp = np.zeros(ntimes)

    for i in range(ntimes):
        lower_limit = (np.random.randn() - 0.5) * np.pi
        w = 2*np.pi * np.random.rand()
        b = (w+1)**(1/(ngrid-1))
        upper_limit = lower_limit + w

        # x[i] = b**i - 1 + lower_limit
        # x[N-1] = upper_limit
        x = b**(np.arange(ngrid)) - 1 + lower_limit
        y = np.sin(x)
        val_ref = np.cos(lower_limit) - np.cos(upper_limit)

        h = np.diff(x)

        val_scipy = simpson(y, x)
        val_isimp = simp2(y, x)
        val_msimp = simpmix2(y, h)

        err_scipy[i] = abs(val_scipy - val_ref)
        err_isimp[i] = abs(val_isimp - val_ref)
        err_msimp[i] = abs(val_msimp - val_ref)

    print('scipy/simp/simpmix= %i/%i/%i'%( np.sum( (err_scipy < err_isimp) & (err_scipy < err_msimp) ), \
                                           np.sum( (err_isimp < err_scipy) & (err_isimp < err_msimp) ), \
                                           np.sum( (err_msimp < err_scipy) & (err_msimp < err_isimp) ) ))

    print('errmax_scipy = %6.4e      errstd_scipy = %6.4e'%(np.max(err_scipy), np.std(err_scipy))) 
    print('errmax_isimp = %6.4e      errstd_isimp = %6.4e'%(np.max(err_isimp), np.std(err_isimp)))
    print('errmax_msimp = %6.4e      errstd_msimp = %6.4e'%(np.max(err_msimp), np.std(err_msimp)))

#bench2(234, 10000)


lower_limit = 0
w = 1.1
ngrid = 6561
b = (w+1)**(1/(ngrid-1))
upper_limit = lower_limit + w

# x[i] = b**i - 1 + lower_limit
# x[N-1] = upper_limit
x = b**(np.arange(ngrid)) - 1 + lower_limit
y = np.sin(x)

h = np.diff(x)

print('%20.12f'%(simpmix2(y,h)))

ref = np.cos(lower_limit) - np.cos(lower_limit+w)
print('%6.4e'%(abs(ref-simpmix2(y,h))))
print('%6.4e'%(abs(ref-simpson(y,x))))

print(w * np.max(h)**4)



