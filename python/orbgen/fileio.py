import re
from itertools import accumulate

def _write_header(f, elem, ecut, rcut, nzeta, nr, dr):
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
            Energy cutoff.
        rcut : int or float
            Cutoff radius.
        nzeta : list of int
            Number of orbitals for each angular momentum.
        lmax : int
            Maximum angular momentum.
        nr : int
            Number of radial grid points.
        dr : float
            Grid spacing.

    '''
    lmax = len(nzeta)-1
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


def _write_chi(f, l, zeta, chi):
    '''
    Writes a numerical radial function to a file object.
    
    Parameters
    ----------
        f : file object
            Must be opened in advance.
        l : int
            Angular momentum.
        zeta : int
            Zeta number.
        chi : list of float
            A numerical radial function.

    '''
    f.write('                Type                   L                   N\n')
    f.write('                   0                   {0}                   {1}\n'.format(l, zeta))
    for ir, chi_of_r in enumerate(chi):
        f.write('{: 21.12e}'.format(chi_of_r))
        if ir % 4 == 3 and ir != len(chi)-1:
            f.write('\n')
    f.write('\n')


def write_nao(fname, elem, ecut, rcut, nr, dr, chi):
    '''
    Generates a numerical atomic orbital file of the ABACUS format.
    
    Parameters
    ----------
        fname : str
            Name/path of the orbital file.
        elem : str
            Element symbol.
        ecut : float
            Energy cutoff.
        rcut : float
            Cutoff radius.
        nr : int
            Number of radial grid points.
        dr : float
            Grid spacing.
        chi : list of list of list of float
            Numerical radial functions as chi[l][zeta][ir].

    '''
    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]
    
    with open(fname, 'w') as f:
        _write_header(f, elem, ecut, rcut, nzeta, nr, dr)
        for l in range(lmax+1):
            for zeta in range(nzeta[l]):
                _write_chi(f, l, zeta, chi[l][zeta])


def read_nao(fname):
    '''
    Reads a numerical atomic orbital file of the ABACUS format.
    
    Parameters
    ----------
        fname : str
            Name/path of the orbital file.
    
    Returns
    -------
        A dictionary containing the following key-value pairs:

        'elem' : str
            Element symbol.
        'ecut' : float
            Energy cutoff.
        'rcut' : float
            Cutoff radius of the orbital.
        'nr' : int
            Number of radial grid points.
        'dr' : float
            Grid spacing.
        'chi' : list of list of list of float
            Numerical radial functions as chi[l][zeta][ir].

    '''
    with open(fname, 'r') as f:
        data = list(filter(None, re.split('\t| |\n', f.read())))

    elem = data[data.index('Element')+1]
    ecut = float(data[data.index('Cutoff(Ry)')+1])
    rcut = float(data[data.index('Cutoff(a.u.)')+1])
    lmax = int(data[data.index('Lmax')+1])

    symbol = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    nzeta = [ int(data[data.index(symbol[l] + 'orbital-->') + 1]) for l in range(lmax+1) ]

    nr = int(data[data.index('Mesh')+1])
    dr = float(data[data.index('dr')+1])

    delim = [i for i, x in enumerate(data) if x == 'Type'] + [len(data)]
    nzeta_cumu = [0] + list(accumulate(nzeta))
    iorb = lambda l, zeta : nzeta_cumu[l] + zeta
    chi = [[ list(map(float, data[delim[iorb(l,zeta)]+6:delim[iorb(l,zeta)+1]] )) \
            for zeta in range(nzeta[l])] for l in range(lmax+1)]

    return {'elem': elem, 'ecut': ecut, 'rcut': rcut, 'nr': nr, 'dr': dr, 'chi': chi}


def _extract(keyword, data):
    '''
    Extracts VALUE from the pattern KEYWORD=" VALUE ".

    '''
    result = re.search(keyword + '=" *([^= ]*) *"', data)
    return result.group(1) if result else None


def read_param(fname):
    '''
    Reads an orbital parameter file of the SIAB/PTG format.
    
    Parameters
    ----------
        fname : str
            Name of the coefficient file to be read.
    
    Returns
    -------
        coeff : list of list of list of float
            Spherical Bessel coefficients as coeff[l][zeta][iq].
        rcut : float
            Cutoff radius of the orbital.
        sigma : float
            Smoothing width.
        symbol : str
            Element symbol.

    '''
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
    rcut = float(_extract('rcut', data))
    sigma = float(_extract('sigma', data))
    symbol = _extract('element', data)

    # split the data into a list of strings
    data = list(filter(None, re.split('\t| ', data)))
    delim = [i for i, x in enumerate(data) if x == 'Type'] + [len(data)]
    ll = [int(data[delim[i]+4]) for i in range(len(delim)-1)]
    lmax = max(ll)
    nzeta = [ll.count(l) for l in range(lmax+1)]
    
    nzeta_cumu = [0] + list(accumulate(nzeta))
    iorb = lambda l, zeta : nzeta_cumu[l] + zeta
    coeff = [[ list(map(float, data[delim[iorb(l,zeta)]+6:delim[iorb(l,zeta)+1]])) \
            for zeta in range(nzeta[l])] for l in range(lmax+1)]
    return coeff, rcut, sigma, symbol


'''
Writes spherical Bessel coefficients.

Parameters
----------
    fname : str
        Name of the coefficient file to be (over)written.
    coeff : list of list of list of float
        Spherical Bessel coefficients as coeff[l][zeta][iq].
    rcut : float
        Cutoff radius of the orbital.
    sigma : float
        Smoothing width.
    elem : str
        Element symbol.
'''
def write_param(fname, coeff, rcut, sigma, elem):
    with open(fname, 'w') as f:
        lmax = len(coeff)-1
        nzeta = [len(coeff[l]) for l in range(lmax+1)]
        n = sum(nzeta)
        f.write('<Coefficient rcut="{0}" sigma="{1}" element="{2}">\n'.format(rcut, sigma, elem))
        f.write('     {0} Total number of radial orbitals.\n'.format(n))
        for l in range(lmax+1):
            for zeta in range(nzeta[l]):
                f.write('    Type   L   Zeta-Orbital\n')
                f.write('      {elem}   {angmom}       {zeta}\n'.format(elem=elem, angmom=l, zeta=zeta))
                for i in range(len(coeff[l][zeta])):
                    f.write('{: 21.14f}\n'.format(coeff[l][zeta][i]))
        f.write('</Coefficient>\n')

############################################################
#                           Test
############################################################
import os
import unittest

class TestFileio(unittest.TestCase):
    def test_read_param(self):
        coeff, rcut, sigma, symbol = read_param('./testfiles/ORBITAL_RESULTS.txt')
    
        lmax = len(coeff)-1
        nzeta = [len(coeff[l]) for l in range(lmax+1)]
        nq = [len(coeff[l][zeta]) for l in range(lmax+1) for zeta in range(nzeta[l])]

        self.assertEqual(symbol, 'In')
        self.assertEqual(rcut, 7.0)
        self.assertEqual(sigma, 0.1)
        self.assertEqual(lmax, 3)
        self.assertEqual(nzeta, [2, 2, 2, 1])
        self.assertEqual(nq, [31] * 7)
        self.assertEqual(coeff[0][0][0], 0.09780237320580)
        self.assertEqual(coeff[0][0][30], 0.00021711814077)
        self.assertEqual(coeff[1][1][0], -0.78111126700600)
        self.assertEqual(coeff[3][0][30], -0.09444436877182)

    def test_read_param(self):
        coeff, rcut, sigma, symbol = read_param('./testfiles/ORBITAL_RESULTS.txt')
    
        lmax = len(coeff)-1
        nzeta = [len(coeff[l]) for l in range(lmax+1)]
        nq = [len(coeff[l][zeta]) for l in range(lmax+1) for zeta in range(nzeta[l])]
    
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

def test_write_param():
    print('Testing write_param...')

    data = read_param('./testfiles/ORBITAL_RESULTS.txt')
    tmpfile = './testfiles/ORBITAL_RESULTS.txt.tmp'
    write_param(tmpfile, *data)
    data2 = read_param(tmpfile)

    assert data == data2

    os.remove(tmpfile)
    print('...Passed!')


def test_read_nao():
    print('Testing read_nao...')
    elem, ecut, rcut, nr, dr, chi = read_nao('./testfiles/C_gga_8au_100Ry_2s2p1d.orb')

    assert elem == 'C'
    assert ecut == 100.0
    assert rcut == 8.0
    assert nr == 801
    assert dr == 0.01

    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]
    nr = [len(chi[l][zeta]) for l in range(lmax+1) for zeta in range(nzeta[l])]
    assert lmax == 2
    assert nzeta == [2, 2, 1]
    assert nr == [801] * 5

    assert chi[0][0][0] == 5.368426038998e-01
    assert chi[0][0][800] == 0.000000000000e+00
    assert chi[0][1][0] == -6.134205291735e-02
    assert chi[1][1][799] == -2.773536551465e-06

    print('...Passed!')


def test_write_nao():
    print('Testing write_nao...')

    data = read_nao('./testfiles/C_gga_8au_100Ry_2s2p1d.orb')
    tmpfile = './testfiles/C_gga_8au_100Ry_2s2p1d.orb.tmp'
    write_nao(tmpfile, *data)
    data2 = read_nao(tmpfile)

    assert data == data2

    os.remove(tmpfile)
    print('...Passed!')


if __name__ == '__main__':
    unittest.main()

    test_read_param()
    test_write_param()
    test_read_nao()
    test_write_nao()

