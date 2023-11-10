'''
Converts a 1D array of (spherical Bessel) coefficients into a nested list.

The 1D array is assumed to be a direct concatenation of spherical Bessel coefficients
for different orbitals. The nested list is a list of lists of lists, where the first
index is the angular momentum, the second index is the zeta number, and the third index
is the wave vector number.

This is the inverse of list2array.

Parameters
----------
    c : 1D array
        A 1D array containing the coefficients of spherical Bessel functions.
    lmax : int
        Maximum angular momentum.
    nzeta : list of int
        Number of orbitals for each angular momentum.
    nq : list of int
        Number of spherical Bessel wave numbers involved in each orbital.
        This list should not contain any 0.
'''
def array2list(c, lmax, nzeta, nq):
    from itertools import accumulate

    assert len(nzeta) == lmax+1
    assert sum(nq) == len(c)

    iq = [0] + list(accumulate(nq))
    iz = [0] + list(accumulate(nzeta))
    return [[list(c[iq[i]:iq[i+1]]) for i in range(iz[l], iz[l+1])] for l in range(lmax+1)]


'''
Flattens a nested list of coefficients into a 1D array.

This is the inverse of array2list.
'''
def list2array(coeff):
    from numpy import array
    return array([clzq for cl in coeff for clz in cl for clzq in clz])


'''
Merges two sets of spherical Bessel coefficients.
'''
def merge(coeff, extra_coeff):
    from copy import deepcopy
    coeff_merged = deepcopy(coeff)

    # pads with empty lists if extra_coeff is longer
    coeff_merged += [[] for _ in range(len(extra_coeff)-len(coeff))]

    # appends extra_coeff to coeff for each angular momentum
    for l, cl in enumerate(extra_coeff):
        coeff_merged[l] += cl

    return coeff_merged


############################################################
#                       Testing
############################################################
def test_merge():
    print('Testing merge...')
    coeff1 = [[[0,0], [1,1]], [], [[3,3,3]]]
    coeff2 = [[], [[2,2,2]], [[4,4]], [[5,5,5]]]

    coeff_ref = [[[0,0],[1,1]], [[2,2,2]], [[3,3,3],[4,4]], [[5,5,5]]]
    coeff_merged = merge(coeff1, coeff2)

    assert coeff_ref == coeff_merged
    print('...Passed!')


if __name__ == '__main__':
    test_merge()

