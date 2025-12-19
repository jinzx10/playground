import sympy as sp

# Define spherical harmonic functions
l, m, theta, phi = sp.symbols('l m theta phi')

# The general form of spherical harmonics is:
Y_lm = sp.Ynm(l, m, theta, phi)

# Set a specific l and m, for example, l = 2, m = 1
Y_lm_value = Y_lm.subs({l: 2, m: 1})

val_theta = 0.123
val_phi = 0.345

# To evaluate Y_lm for specific angles (in radians), you can use .subs() to substitute values
Y_lm_value_at_angles = Y_lm_value.subs({theta: val_theta, phi: val_phi})

# Convert to arbitrary precision
Y_lm_precise = sp.N(Y_lm_value_at_angles, 30)

print(f"Spherical Harmonic Y_2^1 evaluated at θ = {val_theta}, φ = {val_phi} with arbitrary precision:")
print(Y_lm_precise)
#print(Y_lm_value_at_angles.evalf())
print(Y_lm_value_at_angles)
