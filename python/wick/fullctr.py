import numpy as np
from itertools import combinations, permutations

def fullctr(idx_orb, plus_minus = [], dagger = []):
    """
    output = fullctr(idx_orb, plus_minus, dagger) returns all non-zero full contractions.

    Parameters
    ----------
    idx_orb : string
        Orbital indices. Each element must be a letter.
    plus_minus : object convertible to numpy int array
        Orbital occupation info. Default: all zeros.
        Elements must be either 1 (virtual), -1 (occupied) or 0 (arbitrary).
    dagger : object convertible to numpy int array
        Operator dagger info. Default: alternating 1 and -1 (start with 1).
        Elements must be either 1 (dagger) or -1 (no dagger) 
    
    """

    # check idx_orb
    if not isinstance(idx_orb, str) or not idx_orb.isalpha():
        raise Exception("orbital indices '{}' is not a string of letters.".format(idx_orb))

    plus_minus = np.array(plus_minus, dtype=int).flatten()
    dagger = np.array(dagger, dtype=int).flatten()

    # check plus_minus
    if not ( (plus_minus == 1) | (plus_minus == 0) | (plus_minus == -1) ).all():
        raise Exception("plus-minus info {} contains invalid values.".format(plus_minus))

    # check dagger
    if not ( (dagger == 1)  | (dagger == -1) ).all():
        raise Exception("dagger info {} contains invalid values.".format(dagger))

    # if plus_minus is empty, set it to all zeros (arbitrary orbitals)
    if not plus_minus.size:
        plus_minus = np.zeros(len(idx_orb), dtype='int')
    
    # if dagger is empty, set it to alternating 1 and -1 (start with 1)
    if not dagger.size:
        dagger = (-1) ** np.arange(0, len(idx_orb), dtype='int')

    if not ( (len(idx_orb) == len(plus_minus)) and (len(idx_orb) == len(dagger)) ):
        raise Exception("Inconsistent input length.")

    # if the number of operators is odd, result is zero
    if len(idx_orb) % 2:
        return []

    # raw actual creation/annihilation w.r.t. the ref state
    # creation = 1, annihilation = -1, uncertain = 0
    ac_raw = plus_minus * dagger

    # number of raw actual creation/annihilation/uncertain operators
    num_a_raw = sum(ac_raw == -1)
    num_c_raw = sum(ac_raw == 1)
    num_u_raw = sum(ac_raw == 0)

    # if creation/annihilation cannot be fully paired, result is zero
    if num_u_raw < abs(num_a_raw - num_c_raw):
        return []

    # output is a list of strings
    output = []

    # number of extra actual annihilation operatos required to balance
    # actual creation/annihilation
    num_a_extra = (num_u_raw + num_c_raw - num_a_raw) // 2

    # all possible combinations of additional annihilation operators
    list_idx_a_extra = np.array(list(combinations(np.where(ac_raw == 0)[0], num_a_extra)))

    # complete annihilation/creation index list
    list_ac = np.tile(ac_raw, (np.size(list_idx_a_extra,0), 1))
    for i in range(0, np.size(list_ac,0)):
        list_ac[i, list_idx_a_extra[i,:]] = -1
    list_ac[list_ac == 0] = 1

    # for each possible combination of annihilation/creation
    for i in range(0, np.size(list_ac,0)):
        ac = list_ac[i,:]

        # consider all possible contractions
        idx_a = np.where(ac == -1)[0]
        idx_c_all = np.array(list(permutations(np.where(ac == 1)[0])))

        # all 'a' must appear before its corresponding 'c'
        idx_c_all = idx_c_all[(idx_a < idx_c_all).all(axis=1), :]

        # all contractions must happen between a daggered and an undaggered operator
        idx_c_all = idx_c_all[(dagger[idx_a]-dagger[idx_c_all]).all(axis=1), :]
        
        # add signs and collect outputs
        for j in range(0, np.size(idx_c_all,0)):
            output.append(conv2str(ctrsgn(idx_a, idx_c_all[j,:]), idx_a, idx_c_all[j,:], idx_orb, ac*dagger))

    return output


# contraction sign
def ctrsgn(idx_a, idx_c):
    I = np.eye(idx_a.size + idx_c.size)
    perm = np.array([idx_a, idx_c], dtype=int).flatten('F')
    return np.linalg.det(I[:, perm])


# generate contraction string
def conv2str(sgn, idx_a, idx_c, orb, pm):
    pm_char = np.array(['+'] * len(pm))
    pm_char[np.where(pm < 0)] = '-'
    ctrstr = (sgn > 0) * '+' + (sgn < 0) * '-'

    for i in range(0, len(idx_a)):
        ctrstr += orb[idx_a[i]] + orb[idx_c[i]] + '(' + pm_char[idx_a[i]] + ')'

    return ctrstr

