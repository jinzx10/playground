from sympy import *

F2E = Matrix([
    [1, -10, -33, -286],
    [0, Rational(70, 9), Rational(231, 9), Rational(2002, 9)],
    [0, Rational(1, 9), Rational(-1, 3), Rational(7, 9)],
    [0, Rational(5, 3), 2, -Rational(91, 3)],
    ])


A = Matrix([
    [Rational(2,45), Rational(1,33), Rational(50, 1287)],
    [Rational(1, 9), Rational(-1, 3), Rational(7, 9)],
    [Rational(5, 3), 2, -Rational(91, 3)],
    ])

print(A.inv())

QE = Matrix([
    [Rational(225,54), Rational(32175,42), Rational(2475, 42)],
    [11, Rational(-141570, 77), Rational(4356, 77)],
    [Rational(736164, 59400), Rational(368082, 660), -Rational(11154, 100)],
    ])

print(QE.inv())
