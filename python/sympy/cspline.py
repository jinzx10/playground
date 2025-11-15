from sympy import symbols, Matrix

r, d = symbols('r d')

v0 = Matrix([[1, r*d, (r*d)**2, (r*d)**3]])
v1 = Matrix([[0, 1, 2*r*d, 3*(r*d)**2]])
v2 = Matrix([[0, 0, 2, 6*(r*d)]])
v3 = Matrix([[0, 0, 0, 6]])
m = Matrix(
        [[1, 0, 0, 0],
         [1, d, d**2, d**3],
         [0, 0, 2, 0],
         [0, 0, 2, 6*d]
         ])

print(v0 @ m.inv())
print(v1 @ m.inv())
print(v2 @ m.inv())
print(v3 @ m.inv())
