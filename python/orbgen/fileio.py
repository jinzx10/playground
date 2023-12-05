import re
import os
from itertools import accumulate

'''
Writes an ABACUS orbital file header to a file object.

A typical header looks like

<<<<<<< starts here (taken from C_gga_8au_100Ry_2s2p1d.orb)
---------------------------------------------------------------------------
Element                     C
Energy Cutoff(Ry)          100
Radius Cutoff(a.u.)         8
Lmax                        2
Number of Sorbital-->       2
Number of Porbital-->       2
Number of Dorbital-->       1
---------------------------------------------------------------------------
SUMMARY  END

Mesh                        801
dr                          0.01
>>>>>>> ends here

Parameters
----------
    f : file object
        Must be opened in advance.
    elem : str
        Element symbol.
    ecut : int or float
        Energy cutoff. (To be studied...)
    rcut : int or float
        Cutoff radius.
    nzeta : list of int
        Number of orbitals for each angular momentum.
    lmax : int
        Maximum angular momentum.
    dr : float
        Grid spacing.

'''
def write_header(f, elem, ecut, rcut, nzeta, dr):

    lmax = len(nzeta)-1
    nr = int(rcut/dr) + 1
    symbol = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    f.write('---------------------------------------------------------------------------\n')
    f.write('Element                     {0}\n'.format(elem))
    f.write('Energy Cutoff(Ry)           {0}\n'.format(ecut))
    f.write('Radius Cutoff(a.u.)         {0}\n'.format(rcut))
    f.write('Lmax                        {0}\n'.format(lmax))

    for l in range(lmax+1):
        f.write("Number of {0}orbital-->       {1}\n".format(symbol[l], nzeta[l]))

    f.write('---------------------------------------------------------------------------\n')
    f.write('SUMMARY  END\n\n')
    f.write('Mesh                        {0}\n'.format(nr))
    f.write('dr                          {0}\n'.format(dr))


'''
Writes a numerical radial function to a file object.

Parameters
----------
    f : file object
        Must be opened in advance.
    l : int
        Angular momentum.
    izeta : int
        Zeta number.
    chi : list of float
        Numerical radial function.

'''
def write_chi(f, l, izeta, chi):
    f.write('                Type                   L                   N\n')
    f.write('                   0                   {0}                   {1}\n'.format(l, izeta))
    for ir, chi_of_r in enumerate(chi):
        f.write('{: 21.12e}'.format(chi_of_r))
        if ir % 4 == 3 and ir != len(chi)-1:
            f.write('\n')
    f.write('\n')


'''
Generates a numerical atomic orbital file in the ABACUS format.

Parameters
----------
    fname : str
        Name of the orbital file to be generated.
    elem : str
        Element symbol.
    rcut : float
        Cutoff radius of the orbital.
    chi : list of list of list of float
        Numerical radial functions as chi[l][izeta][ir].
    dr : float
        Grid spacing.
'''
def write_orbfile(fname, elem, rcut, chi, dr):
    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]
    
    with open(fname, 'w') as f:
        write_header(f, elem, 100, rcut, nzeta, dr)
        for l in range(lmax+1):
            for izeta in range(nzeta[l]):
                write_chi(f, l, izeta, chi[l][izeta])


'''
Reads an ABACUS orbital file.

Parameters
----------
    fname : str
        Name of the orbital file to be read.

Returns
-------
    elem : str
        Element symbol.
    rcut : float
        Cutoff radius of the orbital.
    chi : list of list of list of float
        Numerical radial functions as chi[l][izeta][ir].
    dr : float
        Grid spacing.
'''
def read_orbfile(fname):
    with open(fname, 'r') as f:
        data = list(filter(None, re.split('\t| |\n', f.read())))

    elem = data[data.index('Element')+1]
    #ecut = float(data[data.index('Cutoff(Ry)')+1])
    rcut = float(data[data.index('Cutoff(a.u.)')+1])
    lmax = int(data[data.index('Lmax')+1])

    symbol = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    nzeta = [ int(data[data.index(symbol[l] + 'orbital-->') + 1]) for l in range(lmax+1) ]

    #nr = int(data[data.index('Mesh')+1])
    dr = float(data[data.index('dr')+1])

    delim = [i for i, x in enumerate(data) if x == 'Type'] + [len(data)]
    nzeta_cumu = [0] + list(accumulate(nzeta))
    iorb = lambda l, izeta : nzeta_cumu[l] + izeta
    chi = [[ list(map(float, data[delim[iorb(l,izeta)]+6:delim[iorb(l,izeta)+1]] )) \
            for izeta in range(nzeta[l])] for l in range(lmax+1)]

    return elem, rcut, chi, dr


'''
Extracts VALUE from the pattern KEYWORD=" VALUE ".
'''
def extract(keyword, data):
    result = re.search(keyword + '=" *([^= ]*) *"', data)
    return result.group(1) if result else None


'''
Reads spherical Bessel coefficients.

Parameters
----------
    fname : str
        Name of the coefficient file to be read.

Returns
-------
    coeff : list of list of list of float
        Spherical Bessel coefficients as coeff[l][izeta][iq].
    rcut : float
        Cutoff radius of the orbital.
    sigma : float
        Smoothing width.
    symbol : str
        Element symbol.
'''
def read_coeff(fname):
    with open(fname, 'r') as f:
        data = f.read()

    # convert '\n' to ' ' for regex matching (.)
    data = data.replace('\n', ' ')

    # extract the Coefficient block
    result = re.search('<Coefficient(.*)</Coefficient>', data)
    if result is None:
        raise ValueError('Coefficient block not found.')
    data = result.group(1)

    # extract the parameters in header
    rcut = float(extract('rcut', data))
    sigma = float(extract('sigma', data))
    symbol = extract('element', data)

    # split the data into a list of strings
    data = list(filter(None, re.split('\t| ', data)))
    delim = [i for i, x in enumerate(data) if x == 'Type'] + [len(data)]
    ll = [int(data[delim[i]+4]) for i in range(len(delim)-1)]
    lmax = max(ll)
    nzeta = [ll.count(l) for l in range(lmax+1)]
    
    nzeta_cumu = [0] + list(accumulate(nzeta))
    iorb = lambda l, izeta : nzeta_cumu[l] + izeta
    coeff = [[ list(map(float, data[delim[iorb(l,izeta)]+6:delim[iorb(l,izeta)+1]])) \
            for izeta in range(nzeta[l])] for l in range(lmax+1)]
    return coeff, rcut, sigma, symbol


'''
Writes spherical Bessel coefficients.

Parameters
----------
    fname : str
        Name of the coefficient file to be (over)written.
    coeff : list of list of list of float
        Spherical Bessel coefficients as coeff[l][izeta][iq].
    rcut : float
        Cutoff radius of the orbital.
    sigma : float
        Smoothing width.
    elem : str
        Element symbol.
'''
def write_coeff(fname, coeff, rcut, sigma, elem):
    with open(fname, 'w') as f:
        lmax = len(coeff)-1
        nzeta = [len(coeff[l]) for l in range(lmax+1)]
        n = sum(nzeta)
        f.write('<Coefficient rcut="{0}" sigma="{1}" element="{2}">\n'.format(rcut, sigma, elem))
        f.write('     {0} Total number of radial orbitals.\n'.format(n))
        for l in range(lmax+1):
            for izeta in range(nzeta[l]):
                f.write('    Type   L   Zeta-Orbital\n')
                f.write('      {elem}   {angmom}       {izeta}\n'.format(elem=elem, angmom=l, izeta=izeta))
                for i in range(len(coeff[l][izeta])):
                    f.write('{: 21.14f}\n'.format(coeff[l][izeta][i]))
        f.write('</Coefficient>\n')

############################################################
#                       Testing
############################################################
def test_read_coeff():
    print('Testing read_coeff...')
    coeff, rcut, sigma, symbol = read_coeff('./testfiles/ORBITAL_RESULTS.txt')

    lmax = len(coeff)-1
    nzeta = [len(coeff[l]) for l in range(lmax+1)]
    nq = [len(coeff[l][izeta]) for l in range(lmax+1) for izeta in range(nzeta[l])]

    assert symbol == 'In'
    assert rcut == 7.0
    assert sigma == 0.1
    assert lmax == 3
    assert nzeta == [2, 2, 2, 1]
    assert nq == [31] * 7
    assert coeff[0][0][0] == 0.09780237320580
    assert coeff[0][0][30] == 0.00021711814077
    assert coeff[1][1][0] == -0.78111126700600
    assert coeff[3][0][30] == -0.09444436877182

    print('...Passed!')


def test_write_coeff():
    print('Testing write_coeff...')

    data = read_coeff('./testfiles/ORBITAL_RESULTS.txt')
    tmpfile = './testfiles/ORBITAL_RESULTS.txt.tmp'
    write_coeff(tmpfile, *data)
    data2 = read_coeff(tmpfile)

    assert data == data2

    os.remove(tmpfile)
    print('...Passed!')


def test_read_orbfile():
    print('Testing read_orbfile...')
    elem, rcut, chi, dr = read_orbfile('./testfiles/C_gga_8au_100Ry_2s2p1d.orb')

    assert elem == 'C'
    assert rcut == 8.0
    assert dr == 0.01

    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]
    nr = [len(chi[l][izeta]) for l in range(lmax+1) for izeta in range(nzeta[l])]
    assert lmax == 2
    assert nzeta == [2, 2, 1]
    assert nr == [801] * 5

    assert chi[0][0][0] == 5.368426038998e-01
    assert chi[0][0][800] == 0.000000000000e+00
    assert chi[0][1][0] == -6.134205291735e-02
    assert chi[1][1][799] == -2.773536551465e-06

    print('...Passed!')


def test_write_orbfile():
    print('Testing write_orbfile...')

    data = read_orbfile('./testfiles/C_gga_8au_100Ry_2s2p1d.orb')
    tmpfile = './testfiles/C_gga_8au_100Ry_2s2p1d.orb.tmp'
    write_orbfile(tmpfile, *data)
    data2 = read_orbfile(tmpfile)

    assert data == data2

    os.remove(tmpfile)
    print('...Passed!')


if __name__ == '__main__':
    test_read_coeff()
    test_write_coeff()
    test_read_orbfile()
    test_write_orbfile()

