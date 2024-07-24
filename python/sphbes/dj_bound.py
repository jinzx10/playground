import numpy as np
from scipy.special import spherical_jn

import matplotlib.pyplot as plt

l = 0
dr = 0.01
n = 5000
r = np.arange(n) * dr
djl = spherical_jn(l, r, derivative=True)

print(np.max(djl))

plt.plot(r, djl)
plt.show()
