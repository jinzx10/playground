import numpy as np
from scipy.interpolate import CubicSpline

f = lambda x: x**3
df = lambda x: 3*x**2
d2f = lambda x: 6*x

x = np.linspace(0.1, 10, 100)
y = f(x)

cs = CubicSpline(x, y, bc_type=((1, df(x[0])), (1, df(x[-1]))))
for xi in x:
    print('y=%12.8e   y_ref=%12.8e   diff=%8.5e'%(cs(xi, 1), df(xi), cs(xi, 1) - df(xi)))
