import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu, sigma):
    return np.exp(-0.5*((x - mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

x = np.linspace(-6, 6, 1000)
y = gauss(x, -2, 1) + gauss(x, 2, 1)

fig, ax = plt.subplots(1, 3, figsize=(15, 4), layout='tight')


p1 = gauss(x, -2, 1)
p2 = gauss(x, 2, 1)

w1 = p1 / (p1+p2)
w2 = p2 / (p1+p2)

y1 = w1 * y
y2 = w2 * y


ax[2].plot(x, y, linewidth=2, color='black')
ax[2].plot(x, y1, linestyle='--', linewidth=2, color='red')
ax[2].plot(x, y2, linestyle='--', linewidth=2, color='orange')


p1 = gauss(x, -2, 3)
p2 = gauss(x, 2, 3)

w1 = p1 / (p1+p2)
w2 = p2 / (p1+p2)

y1 = w1 * y
y2 = w2 * y


ax[1].plot(x, y, linewidth=2, color='black')
ax[1].plot(x, y1, linestyle='--', linewidth=2, color='red')
ax[1].plot(x, y2, linestyle='--', linewidth=2, color='orange')


p1 = np.heaviside(-x, 0.5)
p2 = np.heaviside(x, 0.5)

w1 = p1 / (p1+p2)
w2 = p2 / (p1+p2)

y1 = w1 * y
y2 = w2 * y

ax[0].plot(x, y, linewidth=2, color='black')
ax[0].plot(x, y1, linestyle='--', linewidth=2, color='red')
ax[0].plot(x, y2, linestyle='--', linewidth=2, color='orange')


plt.savefig('schematic_partition.png', dpi=600)
plt.show()
