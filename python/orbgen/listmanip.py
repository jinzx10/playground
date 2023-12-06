from numpy import array
import itertools

'''
Flattens a nested list.
'''
def flatten(x):
    if isinstance(x, list):
        return [elem for i in x for elem in flatten(i)]
    else:
        return [x]


'''
Nests a plain list according to a nesting pattern.

Parameters
----------
    x : list
        A plain list (not nested).
    pattern : list
        A nested list of non-negative int

Examples
--------
>>> x = [1,2,3,4,5]
>>> pattern = [[2], [2], [1]]
>>> print(nest(x, pattern))
[[1, 2], [3, 4], [5]]
>>> pattern = [[2], 2, [1], [0]]
>>> print(nest(x, pattern))
[[1, 2], 3, 4, [5], []]

Notes
-----
Sublists must be specified explicitly in the pattern by enclosing their sizes in lists.
For example, to nest [1,2,3,4,5] into [[1,2],[3,4,5]], one needs a pattern of [[2],[3]];
[2,3] would not do anything.

'''
def nest(x, pattern):
    assert len(x) == sum(flatten(pattern))
    result = []
    idx = 0
    for i in pattern:
        stride = sum(flatten(i))
        if isinstance(i, list):
            result.append(nest(x[idx:idx+stride], i))
        else:
            result += x[idx:idx+stride]
        idx += stride
    return result


'''
Finds the nesting pattern of a nested list.
'''
def pattern(x):
    result = []
    count = 0
    for i, xi in enumerate(x):
        if isinstance(xi, list):
            if count > 0:
                result.append(count)
                count = 0
            result.append(pattern(xi))
        else:
            count += 1
            if i == len(x) - 1:
                result.append(count)

    return result


'''
Converts a 1D array of (spherical Bessel) coefficients into a nested list.

The 1D array is assumed to be a direct concatenation of spherical Bessel coefficients
of different orbitals. The nested list is a list of list of list of float, where the first,
second and third indices label the angular momentum, zeta number, and the wave number, respectively.

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

    assert len(nzeta) == lmax+1
    assert sum(nq) == len(c)

    iq = [0] + list(itertools.accumulate(nq))
    iz = [0] + list(itertools.accumulate(nzeta))
    return [[list(c[iq[i]:iq[i+1]]) for i in range(iz[l], iz[l+1])] for l in range(lmax+1)]


'''
Converts a plain list of spherical Bessel coefficients into a nested list.

The plain list is assumed to be a direct concatenation of spherical Bessel coefficients from different orbitals.
On exit, a nested list (list of list of list of float) will be returned, where the first, second and third
indices label the angular momentum, zeta number and wave number, respectively.

Parameters
----------
    c : list
        A plain list containing the coefficients of spherical Bessel functions.
    nq : list of list of int
        nq[l][izeta] should contain the number of spherical Bessel wave numbers of orbital (l,izeta).
'''
def array2list2(c, nq_nested, nq_flat=None, lmax=None, nzeta=None):
    lmax = len(nq) - 1
    nzeta = [len(nq[l]) for l in range(lmax+1)]
    assert sum([sum(nq[l]) for l in range(lmax+1)]) == len(c)

    iq = [0] + list(accumulate([sum(nq[l]) for l in range(lmax+1)]))
    iz = [0] + list(accumulate(nzeta))
    return [[list(c[iq[i]:iq[i+1]]) for i in range(iz[l], iz[l+1])] for l in range(lmax+1)]


#'''
#Flattens a nested list of coefficients into a 1D array.
#
#This is the inverse of array2list.
#'''
#def list2array(coeff):
#    return array([clzq for cl in coeff for clz in cl for clzq in clz])


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
def test_flatten():
    print('Testing flatten...')

    x = [[1,[2,3],[], 4],[[[5]]], [[]]]
    assert flatten(x) == [1,2,3,4,5]

    print('...Passed!')


def test_nest():
    print('Testing nest...')

    x = [0,1,2,3,4,5,6,7,8,9]
    pattern = [[1,[2],[[1]]], [[2],1], [1], 2]
    assert nest(x, pattern) == [[0, [1, 2], [[3]]], [[4, 5], 6], [7], 8, 9]

    print('...Passed!')


def test_pattern():
    print('Testing pattern...')

    x = [[1,2], 3, [[4,5],6], 7]
    p = pattern(x)
    assert pattern(x) == [[2], 1, [[2],1], 1]

    print('...Passed!')


def test_array2list():
    print('Testing array2list...')

    coeff = array([0,0,0,1,1,2,3,3,3,4,4,5])

    coeff_ref = [[[0,0,0],[1,1]], [[2],[3,3,3]], [], [[4,4]], [[5]]]
    coeff_list = array2list(coeff, 4, [2,2,0,1,1], [3,2,1,3,2,1])

    assert coeff_ref == coeff_list
    print('...Passed!')


#def test_list2array():
#    print('Testing list2array...')
#
#    coeff = [[[0,0,0],[1,1]], [[2],[3,3,3]], [], [[4,4]], [[5]]]
#
#    coeff_ref = array([0,0,0,1,1,2,3,3,3,4,4,5])
#    coeff_array = list2array(coeff)
#
#    assert (coeff_ref == coeff_array).all()
#    print('...Passed!')


def test_merge():
    print('Testing merge...')

    coeff1 = [[[0,0], [1,1]], [], [[3,3,3]]]
    coeff2 = [[], [[2,2,2]], [[4,4]], [[5,5,5]]]

    coeff_ref = [[[0,0],[1,1]], [[2,2,2]], [[3,3,3],[4,4]], [[5,5,5]]]
    coeff_merged = merge(coeff1, coeff2)

    assert coeff_ref == coeff_merged
    print('...Passed!')


if __name__ == '__main__':
    test_flatten()
    test_nest()
    test_pattern()
    test_array2list()
    #test_list2array()
    test_merge()


