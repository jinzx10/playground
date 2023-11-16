import numpy as np
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize, dual_annealing, basinhopping

sz = 60
rastrigin = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)

#lb = [-5.12] * sz
#ub = [5.12] * sz
#res = dual_annealing(rastrigin, bounds=list(zip(lb, ub)), maxiter=10000)
#print(res.x, res.fun)
#print(res)
#exit()

#x0 = np.random.rand(sz)
#res = basinhopping(rastrigin, x0)
#print(res.x, res.fun)
#print(res)
#exit()

x0 = np.random.rand(sz) * 0.8
res  = minimize(rastrigin, x0, method='BFGS', options={'disp': True, 'eps': 1e-3, 'gtol': 1e-3})
#res  = minimize(rastrigin, x0, method='Nelder-Mead', options={'disp': True})
print('xopt = ', res.x)
print('fopt = ', res.fun)
print('nfev = ', res.nfev)
print(res)
