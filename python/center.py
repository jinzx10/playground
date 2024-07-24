import numpy as np
from scipy.linalg import qr

def radii(c1, c2, c3):
    '''
    Given the centers of three mutually tangent spheres, return their radii.

    '''
    d12 = np.linalg.norm(c1-c2) # r1+r2
    d23 = np.linalg.norm(c2-c3) # r2+r3
    d31 = np.linalg.norm(c3-c1) # r3+r1

    d = (d12 + d23 + d31)/2
    return d - d23, d - d31, d - d12


def invert(perm):
    '''
    Inverse of a permutation.

    '''
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def center(c1, c2, c3, r0):
    '''
    Given the centers of three mutually tangent spheres and the radius of
    a fourth sphere tangent to all three, return the possible centers of
    the fourth sphere.

    '''

    r1, r2, r3 = radii(c1, c2, c3)

    k1 = (r0+r1)**2 - np.dot(c1,c1)
    k2 = (r0+r2)**2 - np.dot(c2,c2)
    k3 = (r0+r3)**2 - np.dot(c3,c3)
    
    k = np.array([k1-k2, k2-k3, k3-k1]) / 2
    A = np.array([c2-c1, c3-c2, c1-c3])
    # A @ [x0,y0,z0] = k
    
    Q, R, perm = qr(A, pivoting=True) # A[:,perm] = Q @ R
    iperm = invert(perm)

    t = Q.T @ k
    # now we have R[:,iperm] @ c0 = t where c0 = [x0,y0,z0]
    # or R @ c0_ = t where c0_ = c0[perm]

    # since the rank of A is 2, the third row of R must be [0,0,0]

    # express the first and second elements of c0_ in terms of the third
    # for simplicity, denote them x0_, y0_ and z0_ respectively
    p = np.linalg.solve(R[:2,:2], -R[:2,2])
    q = np.linalg.solve(R[:2,:2], t[:2])

    # now we have [x0_, y0_] = p*z0_ + q
    # and thus norm(z0_*[p,1] + [q,0] - c1[perm]) = r0 + r1
    p_ = np.append(p, 1)
    q_ = np.append(q, 0)
    q_c1_ = q_ - c1[perm]

    # solve the quadratic equation a*z0_**2 + b*z0_ + c = 0
    a = np.dot(p_, p_)
    b = 2 * np.dot(p_, q_c1_)
    c = np.dot(q_c1_, q_c1_) - (r0+r1)**2

    D = b**2 - 4*a*c
    if D < 0:
        return ()

    z0_ = np.array([(-b + np.sqrt(D)) / (2*a), (-b - np.sqrt(D)) / (2*a)])
    c0_ = np.outer(p_, z0_) + q_.reshape(3, 1)
    return (c0_[iperm, 0], c0_[iperm, 1]) # permute c0_ = c0[perm] back to c0


################################################################

import unittest

class TestCenter(unittest.TestCase):

    def test_xy_equilateral(self):
        '''
        The centers of the given three spheres form an equilateral triangle
        on the xy-plane.

        '''
        nt = 1000

        c1 = np.array([0, 0, 0])
        c2 = np.array([2, 0, 0])
        c3 = np.array([1, np.sqrt(3), 0])
        r1, r2, r3 = radii(c1, c2, c3)

        for i in range(nt):
            r0 = 2 * np.random.rand()
            c0 = center(c1, c2, c3, r0)
            self.assertEqual(len(c0), 2 if r0 > (2/np.sqrt(3)-1)*r1 else 0)
            for c in c0:
                self.assertAlmostEqual(np.linalg.norm(c-c1), r0+r1, 10)
                self.assertAlmostEqual(np.linalg.norm(c-c2), r0+r2, 10)
                self.assertAlmostEqual(np.linalg.norm(c-c3), r0+r3, 10)


    def test_yz_equilateral(self):
        '''
        The centers of the given three spheres form an equilateral triangle
        on the yz-plane.

        '''
        nt = 1000

        c1 = np.array([0, 0, 0])
        c2 = np.array([0, 2, 0])
        c3 = np.array([0, 1, np.sqrt(3)])
        r1, r2, r3 = radii(c1, c2, c3)

        for i in range(nt):
            r0 = 2 * np.random.rand()
            c0 = center(c1, c2, c3, r0)
            self.assertEqual(len(c0), 2 if r0 > (2/np.sqrt(3)-1)*r1 else 0)
            for c in c0:
                self.assertAlmostEqual(np.linalg.norm(c-c1), r0+r1, 10)
                self.assertAlmostEqual(np.linalg.norm(c-c2), r0+r2, 10)
                self.assertAlmostEqual(np.linalg.norm(c-c3), r0+r3, 10)


    def test_zx_equilateral(self):
        '''
        The centers of the given three spheres form an equilateral triangle
        on the zx-plane.

        '''
        nt = 1000

        c1 = np.array([0, 0, 0])
        c2 = np.array([2, 0, 0])
        c3 = np.array([1, 0, np.sqrt(3)])
        r1, r2, r3 = radii(c1, c2, c3)

        for i in range(nt):
            r0 = 2 * np.random.rand()
            c0 = center(c1, c2, c3, r0)
            self.assertEqual(len(c0), 2 if r0 > (2/np.sqrt(3)-1)*r1 else 0)
            for c in c0:
                self.assertAlmostEqual(np.linalg.norm(c-c1), r0+r1, 10)
                self.assertAlmostEqual(np.linalg.norm(c-c2), r0+r2, 10)
                self.assertAlmostEqual(np.linalg.norm(c-c3), r0+r3, 10)


    def test_random(self):
        pass_count = 0
        skip_count = 0
        nt = 10000

        for i in range(nt):
            c1 = np.random.randn(3)
            c2 = np.random.randn(3)
            c3 = np.random.randn(3)
            r0 = 5 * np.random.rand()

            c0 = center(c1, c2, c3, r0)

            if not c0:
                skip_count += 1
                continue

            r1, r2, r3 = radii(c1, c2, c3)
            for i, c in enumerate(c0):
                self.assertAlmostEqual(np.linalg.norm(c-c1), r0+r1, 12)
                self.assertAlmostEqual(np.linalg.norm(c-c2), r0+r2, 12)
                self.assertAlmostEqual(np.linalg.norm(c-c3), r0+r3, 12)

            pass_count += 1

        print(f'random test: pass/skip/total = {pass_count}/{skip_count}/{nt}')


if __name__ == "__main__":
    unittest.main(module='center')

