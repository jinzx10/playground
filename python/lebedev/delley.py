import numpy as np

# number of reduced points for each type
npoints = [1, 1, 0, 3, 1, 0]

x = [
        0.00000000000000000,
        0.57735026918962576,
        0.18511563534473617,
        0.39568947305594191,
        0.69042104838229218,
        0.47836902881215020,
]

y = [
        0.00000000000000000,
        0.57735026918962576, 
        0.18511563534473617,
        0.39568947305594191,
        0.21595729184584883,
        0.00000000000000000,
]

w = [
        0.0038282704949371616, 
        0.0097937375124875125,
        0.0082117372831911110,
        0.0095954713360709628,
        0.0099428148911781033,
        0.0096949963616630283,
]

# vertex
def group6():
    return [( 1,  0,  0),
            (-1,  0,  0),
            ( 0,  1,  0),
            ( 0, -1,  0),
            ( 0,  0,  1),
            ( 0,  0, -1)
            ]

# face center
def group8():
    a = 1.0 / np.sqrt(3)
    return [( a,  a,  a),
            ( a,  a, -a),
            ( a, -a,  a),
            ( a, -a, -a),
            (-a,  a,  a),
            (-a,  a, -a),
            (-a, -a,  a),
            (-a, -a, -a)
            ]

# edge center
def group12():
    a = 1.0 / np.sqrt(2)
    return [( a,  a,  0),
            ( a, -a,  0),
            (-a,  a,  0),
            (-a, -a,  0),
            ( a,  0,  a),
            ( a,  0, -a),
            (-a,  0,  a),
            (-a,  0, -a),
            ( 0,  a,  a),
            ( 0,  a, -a),
            ( 0, -a,  a),
            ( 0, -a, -a)
            ]

# (u, u, sqrt(1-2u**2)) 
# plus/minus -> 8
# permutation -> 3
def group24a(x, y):
    u = x if np.abs(x-y) < 1e-12 else np.sqrt(1.0 - x**2 - y**2)
    v = np.sqrt(1.0 - 2*u**2)

    return [
            ( u,  u,  v),
            ( u,  u, -v),
            ( u, -u,  v),
            ( u, -u, -v),
            (-u,  u,  v),
            (-u,  u, -v),
            (-u, -u,  v),
            (-u, -u, -v),
            ( v,  u,  u),
            ( v,  u, -u),
            ( v, -u,  u),
            ( v, -u, -u),
            (-v,  u,  u),
            (-v,  u, -u),
            (-v, -u,  u),
            (-v, -u, -u),
            ( u,  v,  u),
            ( u,  v, -u),
            ( u, -v,  u),
            ( u, -v, -u),
            (-u,  v,  u),
            (-u,  v, -u),
            (-u, -v,  u),
            (-u, -v, -u),
            ]

# (u, 0, sqrt(1-u**2))
# plus/minus -> 4
# permutation -> 6
def group24b(x, y):
    u = x if np.abs(x) > 1e-12 else y
    v = np.sqrt(1.0 - u**2)
    return [
            ( u,  0,  v),
            ( u,  0, -v),
            (-u,  0,  v),
            (-u,  0, -v),
            ( v,  0,  u),
            ( v,  0, -u),
            (-v,  0,  u),
            (-v,  0, -u),
            ( 0,  u,  v),
            ( 0,  u, -v),
            ( 0, -u,  v),
            ( 0, -u, -v),
            ( 0,  v,  u),
            ( 0,  v, -u),
            ( 0, -v,  u),
            ( 0, -v, -u),
            ( u,  v,  0),
            ( u, -v,  0),
            (-u,  v,  0),
            (-u, -v,  0),
            ( v,  u,  0),
            ( v, -u,  0),
            (-v,  u,  0),
            (-v, -u,  0),
            ]

# (u, v, sqrt(1-u**2-v**2))
# plus/minus -> 8
# permutation -> 6
def group48(x, y):
    u, v, r = x, y, np.sqrt(1.0 - x**2 - y**2)
    return [
            ( u,  v,  r),
            ( u,  v, -r),
            ( u, -v,  r),
            ( u, -v, -r),
            (-u,  v,  r),
            (-u,  v, -r),
            (-u, -v,  r),
            (-u, -v, -r),
            ( u,  r,  v),
            ( u,  r, -v),
            ( u, -r,  v),
            ( u, -r, -v),
            (-u,  r,  v),
            (-u,  r, -v),
            (-u, -r,  v),
            (-u, -r, -v),
            ( v,  u,  r),
            ( v,  u, -r),
            ( v, -u,  r),
            ( v, -u, -r),
            (-v,  u,  r),
            (-v,  u, -r),
            (-v, -u,  r),
            (-v, -u, -r),
            ( v,  r,  u),
            ( v,  r, -u),
            ( v, -r,  u),
            ( v, -r, -u),
            (-v,  r,  u),
            (-v,  r, -u),
            (-v, -r,  u),
            (-v, -r, -u),
            ( r,  u,  v),
            ( r,  u, -v),
            ( r, -u,  v),
            ( r, -u, -v),
            (-r,  u,  v),
            (-r,  u, -v),
            (-r, -u,  v),
            (-r, -u, -v),
            ( r,  v,  u),
            ( r,  v, -u),
            ( r, -v,  u),
            ( r, -v, -u),
            (-r,  v,  u),
            (-r,  v, -u),
            (-r, -v,  u),
            (-r, -v, -u),
            ]


def gridgen(npoints, x, y, w):
    idx = [0]
    idx.extend(np.cumsum(npoints))
    #print(idx)

    grid = []
    weight = []

    # group 6
    for i in range(npoints[0]):
        grid.extend(group6())
        weight.extend([w[idx[0]]]*6)

    # group 8
    for i in range(npoints[1]):
        grid.extend(group8())
        weight.extend([w[idx[1]]]*8)

    # group 12
    for i in range(npoints[2]):
        grid.extend(group12())
        weight.extend([w[idx[2]]]*12)

    # group 24a
    for i in range(npoints[3]):
        grid.extend(group24a(x[idx[3]+i], y[idx[3]+i]))
        weight.extend([w[idx[3]+i]]*24)

    # group 24b
    for i in range(npoints[4]):
        grid.extend(group24b(x[idx[4]+i], y[idx[4]+i]))
        weight.extend([w[idx[4]+i]]*24)

    # group 48
    for i in range(npoints[5]):
        grid.extend(group48(x[idx[5]+i], y[idx[5]+i]))
        weight.extend([w[idx[5]+i]]*48)

    return grid, weight



# generate some random function from a combination of spherical harmonics
coef = []
l = []
m = []
for i in range(9):
    coef.append(np.random.randn())
    l.append(i)
    m.append(np.random.randint(-l[i], l[i]+1)) # m in [-l, l]


def f2(r): # r is a unit vector
    from scipy.special import sph_harm

    assert(np.abs(np.linalg.norm(r)-1)<1e-12)

    polar = np.arccos(r[2])
    azimuth = np.arctan2(r[1], r[0])

    f = 0.0
    for i, c in enumerate(coef):
        f += c * sph_harm(m[i], l[i], azimuth, polar)

    return np.abs(f)**2

#print(coef)
#print(l)
#print(m)

# assume f = \sum coef[i] * Y[l[i], m[i]]
# \int |f|^2 d\Omega = \sum_i coef[i]**2

# NOTE: scipy's sph_harm is not normalized!
ref = sum([c**2 for c in coef]) / 4 / np.pi
print(f'ref = {ref}')


grid, weight = gridgen(npoints, x, y, w)
#print(len(grid))
#print('sum of weight = ', sum(weight))
#print(weight)

leb = 0

for i, r in enumerate(grid):
    leb += f2(r) * weight[i]

print(f'leb = {leb}')

print('abs(leb - ref) = ', np.abs(leb-ref))










