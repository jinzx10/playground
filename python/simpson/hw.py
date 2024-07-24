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

# irregularly-spaced
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

def simpmix2(f, x):

    n = len(x)
    assert(n >= 2)

    if n == 2:
        return 0.5 * (x[1] - x[0]) * (f[0] + f[1])

    h = np.diff(x)
    if n == 4:
        w = h[0] + h[1] + h[2]
        return w / 12.0 * ( 3.0 + ( h[2]*(h[2]-2.0*h[0])/(h[0]+h[1]) - h[1] ) / h[0] ) * f[0] + \
                w**3 / 12.0 * (h[0]+h[1]-h[2]) / (h[0]*h[1]*(h[1]+h[2])) * f[1] + \
                w**3 / 12.0 * (h[2]+h[1]-h[0]) / (h[2]*h[1]*(h[1]+h[0])) * f[2] + \
                w / 12.0 * ( 3.0 + ( h[0]*(h[0]-2.0*h[2])/(h[2]+h[1]) - h[1] ) / h[2] ) * f[3];

    if n % 2 == 1:
        hodd = h[1::2]
        heven = h[0::2]
        hoe = hodd / heven
        heo = heven / hodd
        return np.sum( (heven+hodd)/6*( (2-hoe)*f[0:-1:2] + (2+hoe+heo)*f[1:-1:2] + (2-heo)*f[2::2]) )
    else:
        return simpmix2(f[:-3], x[:-3]) + simpmix2(f[-4:], x[-4:])


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

    print('simp/simpmix/scipy = %i/%i/%i'%(np.sum( (err_isimp < err_scipy) & (err_isimp < err_msimp) ), \
                                           np.sum( (err_msimp < err_scipy) & (err_msimp < err_isimp) ), \
                                           np.sum( (err_scipy < err_isimp) & (err_scipy < err_msimp) )))
    print('errmax_scipy = %6.4e      errstd_scipy = %6.4e'%(np.max(err_scipy), np.std(err_scipy))) 
    print('errmax_isimp = %6.4e      errstd_isimp = %6.4e'%(np.max(err_isimp), np.std(err_isimp)))
    print('errmax_msimp = %6.4e      errstd_msimp = %6.4e'%(np.max(err_msimp), np.std(err_msimp)))

def bench2(ngrid, ntimes):
    err_scipy = np.zeros(ntimes)
    err_isimp = np.zeros(ntimes)
    err_msimp = np.zeros(ntimes)

    for i in range(ntimes):
        lower_limit = (np.random.randn() - 0.5) * np.pi
        upper_limit = lower_limit + 2*np.pi * np.random.rand()
        
        x = lower_limit + (upper_limit - lower_limit) * np.random.rand(ngrid)
        x = np.sort(x)
        y = np.sin(x)
        val_ref = np.cos(lower_limit) - np.cos(upper_limit)

        val_scipy = simpson(y, x)
        val_isimp = simp2(y, x)
        val_msimp = simpmix2(y, x)

        err_scipy[i] = abs(val_scipy - val_ref)
        err_isimp[i] = abs(val_isimp - val_ref)
        err_msimp[i] = abs(val_msimp - val_ref)

    print('simp/simpmix/scipy = %i/%i/%i'%(np.sum( (err_isimp < err_scipy) & (err_isimp < err_msimp) ), \
                                           np.sum( (err_msimp < err_scipy) & (err_msimp < err_isimp) ), \
                                           np.sum( (err_scipy < err_isimp) & (err_scipy < err_msimp) )))
    print('errmax_scipy = %6.4e      errstd_scipy = %6.4e'%(np.max(err_scipy), np.std(err_scipy))) 
    print('errmax_isimp = %6.4e      errstd_isimp = %6.4e'%(np.max(err_isimp), np.std(err_isimp)))
    print('errmax_msimp = %6.4e      errstd_msimp = %6.4e'%(np.max(err_msimp), np.std(err_msimp)))

#def benchmark(ngrid, ntimes):
#    err_scipy = np.zeros(ntimes)
#    err_isimp = np.zeros(ntimes)
#
#    for i in range(ntimes):
#        lower_limit = (np.random.randn() - 0.5) * np.pi
#        upper_limit = lower_limit + 2*np.pi * np.random.rand()
#        
#        x = np.linspace(lower_limit, upper_limit, ngrid)
#        y = np.sin(x)
#        val_ref = np.cos(lower_limit) - np.cos(upper_limit)
#
#        val_scipy = simpson(y, x)
#        val_isimp = simp(y, x[1]-x[0])
#
#        err_scipy[i] = abs(val_scipy - val_ref)
#        err_isimp[i] = abs(val_isimp - val_ref)
#
#    return np.sum(err_scipy >= err_isimp), np.max(err_scipy), np.max(err_isimp), np.std(err_scipy), np.std(err_isimp)

def benchmark2(ngrid, ntimes):
    err_scipy = np.zeros(ntimes)
    err_isimp = np.zeros(ntimes)

    for i in range(ntimes):
        lower_limit = (np.random.randn() - 0.5) * np.pi
        upper_limit = lower_limit + 2*np.pi * np.random.rand()
        
        x = lower_limit + (upper_limit - lower_limit) * np.random.rand(ngrid)
        x = np.sort(x)
        y = np.sin(x)
        val_ref = np.cos(lower_limit) - np.cos(upper_limit)

        val_scipy = simpson(y, x)
        val_isimp = simp2(y, x)

        err_scipy[i] = abs(val_scipy - val_ref)
        err_isimp[i] = abs(val_isimp - val_ref)

    return np.sum(err_scipy >= err_isimp), np.max(err_scipy), np.max(err_isimp), np.std(err_scipy), np.std(err_isimp)

#A = 0.1
#B = 0.2
#C = 0.3
#D = 0.4
#x0 = 0
#
#f = lambda x: A*(x-x0)**3 + B*(x-x0)**2 + C*(x-x0) + D
#
#x1 = 0.1
#x2 = 0.2
#x3 = 0.4
#x4 = 0.8
#
#h1 = x2-x1
#h2 = x3-x2
#h3 = x4-x3
#
#y1 = f(x1)
#y2 = f(x2)
#y3 = f(x3)
#y4 = f(x4)
#
#
#I = lambda x: A/4*(x-x0)**4 + B/3*(x-x0)**3 + C/2*(x-x0)**2 + D*(x-x0)
#ref = I(x4) - I(x1)
#
#print(ref, simpmix2([y1,y2,y3,y4], [x1,x2,x3,x4]))

#benchmark2(100, 10000)
bench2(500, 20000)

#exit()

#start = 0
#for ngrid in [4,8,16]:
#    end = 0.1
#    while (end < np.pi):
#        dx = (end-start)/(ngrid-1)
#        x = start + np.arange(ngrid) * dx
#        f = np.sin(x)
#        ref = np.cos(start) - np.cos(end)
#        err = (x[-4] - x[0])*dx**4/180 + (x[-1]-x[-4])*dx**4/80
#        print('%6i  %8.4f   %6s    %6s'%(ngrid, end, abs(ref-simp(f, dx))<err, abs(ref-simpmix(f, dx)) < err) )
#        end += 0.1
#
#
#exit()
#n = 5
#x = 0.1 * np.arange(n)
#x = np.exp(x)
#print(simp(np.sin(x), 0.1))
#print(simp2(np.sin(x), x))
#exit()

#ntimes = 20000
#ngrid = 500
#iwin, errmax_scipy, errmax_isimp, errstd_scipy, errstd_isimp = benchmark2(ngrid, ntimes)
#
#print('iwin = %i/%i'%(iwin, ntimes))
#print('errmax_scipy = %6.4e      errstd_scipy = %6.4e'%(errmax_scipy, errstd_scipy))
#print('errmax_isimp = %6.4e      errstd_isimp = %6.4e'%(errmax_isimp, errstd_isimp))


#lower_limit = (np.random.randn() - 0.5) * np.pi
#upper_limit = lower_limit + 2*np.pi * np.random.rand()
#
#ngrid = 108
#x = lower_limit + (upper_limit - lower_limit) * np.random.rand(ngrid)
#x = np.sort(x)
#y = np.sin(x)
#
#val_ref = np.cos(lower_limit) - np.cos(upper_limit)
#val_scipy = simpson(y, x)
#val_simp2 = simp2(y, x)
#
#err_scipy = abs(val_scipy - val_ref)
#err_simp2 = abs(val_simp2 - val_ref)
#
#print('scipy = ', err_scipy)
#print('simp2 = ', err_simp2)



