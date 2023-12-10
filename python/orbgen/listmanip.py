def flatten(x):
    '''
    Flattens a nested list.

    '''
    if isinstance(x, list):
        return [elem for i in x for elem in flatten(i)]
    else:
        return [x]


def nest(x, pattern):
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
    [2,3] would not do anything. Empty lists must also be specified explicitly as [0].
    
    '''
    assert len(pattern) > 0
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


def nestpat(x):
    '''
    Finds the nesting pattern of a list.
    
    The nesting pattern is a (nested) list of non-negative integers.
    For a plain (i.e., not nested) list of length n, the pattern is [n].
    For a nested list like [[1,2], [[]], 3, 4, [5]], the pattern is [[2], [[0]], 2, [1]].
    
    '''
    result = []
    count = 0
    for i, xi in enumerate(x):
        if isinstance(xi, list):
            if count > 0:
                result.append(count)
                count = 0
            result.append(nestpat(xi))
        else:
            count += 1
            if i == len(x) - 1:
                result.append(count)

    return result if len(result) > 0 else [0]


def merge(l1, l2, depth):
    '''
    Merges two (nested) lists at a specified depth.

    A depth-0 merge concatenates two lists.
    A depth-n merge applies depth-(n-1) merge to two lists element-wise.

    '''
    assert isinstance(depth, int) and depth >= 0
    assert isinstance(l1, list) and isinstance(l2, list)

    if depth == 0:
        return l1 + l2

    l_long, l_short = (l1, l2) if len(l1) >= len(l2) else (l2, l1)

    from copy import deepcopy
    l = deepcopy(l_long)
    for i in range(len(l_short)):
        l[i] = merge(l1[i], l2[i], depth-1)

    return l


############################################################
#                       Test
############################################################
import unittest
class TestListManip(unittest.TestCase):

    def test_flatten(self):
        x = [[[]]]
        self.assertEqual(flatten(x), [])
    
        x = [[], [], []]
        self.assertEqual(flatten(x), [])
    
        x = [[[], 1]]
        self.assertEqual(flatten(x), [1])
    
        x = [[1, [2, 3], [], 4], [[[5]]], [[]]]
        self.assertEqual(flatten(x), [1, 2, 3, 4, 5])
    
    
    def test_nest(self):
        x = []
        pattern = [[[0]]]
        self.assertEqual(nest(x, pattern), [[[]]])
    
        x = [7]
        pattern = [[0], [0], [[1]]]
        self.assertEqual(nest(x, pattern), [[], [], [[7]]])
    
        x = [0, 1, 2, 3, 4]
        pattern = [[3], 2, [0]]
        self.assertEqual(nest(x, pattern), [[0, 1, 2], 3, 4, []])
    
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        pattern = [[1, [2], [[1]]], [[2], 1], [1], 2]
        self.assertEqual(nest(x, pattern), [[0, [1, 2], [[3]]], [[4, 5], 6], [7], 8, 9])
    
    
    def test_nestpat(self):
        x = [1, 2, 3]
        self.assertEqual(nestpat(x), [3])
    
        x = [[1, 2], 3, [[4, 5], 6], 7]
        self.assertEqual(nestpat(x), [[2], 1, [[2], 1], 1])
    
        x = [[1, 2], 3, 4, [[]], [5], []]
        self.assertEqual(nestpat(x), [[2], 2, [[0]], [1], [0]])
    
    
    def test_merge(self):
        l1 = []
        l2 = []
        self.assertEqual(merge(l1, l2, 0), [])
    
        l1 = [7]
        l2 = []
        self.assertEqual(merge(l1, l2, 0), [7])
    
        l1 = [0]
        l2 = [1, 2]
        self.assertEqual(merge(l1, l2, 0), [0, 1, 2])
    
        l1 = [[0, 1], [2, 3]]
        l2 = [[], [4, 5], [[6, 7]]]
        self.assertEqual(merge(l1, l2, 0), [[0, 1], [2, 3], [], [4, 5], [[6, 7]]])
        self.assertEqual(merge(l1, l2, 1), [[0, 1], [2, 3, 4, 5], [[6, 7]]])
    
        l1 = [[[0], [1]], [[2, 3]]]
        l2 = [[[4, 5]], [[6], [7]]]
        self.assertEqual(merge(l1, l2, 1), [[[0], [1], [4, 5]], [[2, 3], [6], [7]]])
        self.assertEqual(merge(l1, l2, 2), [[[0, 4, 5], [1]], [[2, 3, 6], [7]]])


if __name__ == '__main__':
    unittest.main()

