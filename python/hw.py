import numpy as np


c1 = np.array([1,1,0])
c2 = np.array([0,1,1])
c3 = np.array([1,0,1])

d12 = np.linalg.norm(c1-c2) # r1+r2
d23 = np.linalg.norm(c2-c3) # r2+r3
d31 = np.linalg.norm(c3-c1) # r3+r1
d = (d12 + d23 + d31)/2

r1 = d - d23
r2 = d - d31
r3 = d - d12

r0 = 2*r1

k1 = (r0+r1)**2 - np.dot(c1,c1)
k2 = (r0+r2)**2 - np.dot(c2,c2)
k3 = (r0+r3)**2 - np.dot(c3,c3)

k = np.array([k1-k2, k2-k3, k3-k1]) / 2

A = np.array([c2-c1, c3-c2, c1-c3])

e1 = A[:,0]
e2 = A[:,1]
e3 = A[:,2]

# at this stage, A @ [x0,y0,z0] = k

f2 = np.cross(np.cross(e1, e3), e2)
f3 = np.cross(np.cross(e1, e2), e3)

# quadratic equation of x0
# a*x0**2 + b*x0 + c = 0
# from (x0-x1)**2 + (x0-x2)**2 + (x0-x3)**2 = (r0+r1)**2

ya = -np.dot(f3,e1) / np.dot(f3,e2)
za = -np.dot(f2,e1) / np.dot(f2,e3)

yb = np.dot(f3, k) / np.dot(f3, e2)
zb = np.dot(f2, k) / np.dot(f2, e3)

a = 1 + ya**2 + za**2
b = -2*c1[0] + 2 * (yb-c1[1]) * ya + 2 * (zb-c1[2]) * za
c = c1[0]**2 + (yb-c1[1])**2 + (zb-c1[2])**2 - (r0+r1)**2

x0 = np.roots([a, b, c])
print(x0) # supposed to be 4/3 and 0



