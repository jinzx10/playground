import numpy as np
import matplotlib.pyplot as plt

k = 1.234
b = 3.4
n = 10

x = np.arange(n)
y = k * x + b + 0.3 * np.random.randn(n)

xbar = x.mean()
ybar = y.mean()

k_fit = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
b_fit = ybar - k_fit * xbar

plt.plot(x, y, 'o')
plt.plot(x, k_fit * x + b_fit)

plt.show()



