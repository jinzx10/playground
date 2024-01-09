def _write_atomic_species(f, species):
    '''
    Writes the ATOMIC_SPECIES block of an ABACUS STRU file.

    '''
    for elem in species:
        elem.setdefault('mass', 1.0)
        elem.setdefault('pp_type', '')

    width = { key + '_w' : max([len(str(elem[key])) for elem in species]) \
            for key in species[0].keys()}

    f.write('ATOMIC_SPECIES\n')
    for elem in species:
        f.write('{symbol:<{symbol_w}}  {mass:>{mass_w}}  {pp_file:>{pp_file_w}}  {pp_type:<{pp_type_w}}\n' \
                .format(**elem, **width))


def _write_numerical_orbital(f, orbitals):
    '''
    Writes the NUMERICAL_ORBITAL block of an ABACUS STRU file.

    '''
    if len(orbitals) > 0:
        f.write('NUMERICAL_ORBITAL\n')
        for orb in orbitals:
            f.write('{}\n'.format(orb))


def _write_lattice(f, lattice):
    '''
    Writes the LATTICE_CONSTANT, LATTICE_PARAMETER and LATTICE_VECTOR blocks of an ABACUS STRU file.

    '''
    f.write('LATTICE_CONSTANT\n')
    f.write('{}\n'.format(lattice['latconst']))

    if 'latvec' in lattice and len(lattice['latvec']) > 0:
        assert len(lattice['latvec']) == 3
        f.write('\n')
        f.write('LATTICE_VECTOR\n')
        for vec in lattice['latvec']:
            f.write('{} {} {}\n'.format(vec[0], vec[1], vec[2]))

    if 'latparam' in lattice and len(lattice['latparam']) > 0:
        f.write('\n')
        f.write('LATTICE_PARAMETER\n')
        for param in lattice['latparam']:
            f.write('{} '.format(param))
        f.write('\n')


def _write_atoms(f, atoms):
    '''
    Writes the ATOMIC_POSTITIONS block of an ABACUS STRU file.

    '''
    coord_type, elements = atoms

    f.write('ATOMIC_POSITIONS\n')
    f.write('{}\n'.format(coord_type))

    for elem, info in elements.items():
        f.write('\n{}\n'.format(elem))
        f.write('{}\n'.format(info['mag_each']))
        f.write('{}\n'.format(info['num']))
        for i in range(info['num']):
            f.write('{} {} {}'.format(info['coord'][i][0], info['coord'][i][1], info['coord'][i][2]))

            # optional extra parameters after the coordinates
            for key in ['m', 'v']:
                if key in info and len(info[key][i]) > 0:
                    f.write(' {} {} {} {}'.format(key, info[key][i][0], info[key][i][1], info[key][i][2]))

            if 'mag' in info and len(info['mag'][i]) > 0:
                mag_coord_type, mag = info['mag'][i]
                assert mag_coord_type in ['Cartesian', 'Spherical']
                if mag_coord_type == 'Cartesian':
                    f.write(' mag {} {} {}'.format(mag[0], mag[1], mag[2]))
                else:
                    f.write(' mag {} angle1 {} angle2 {}'.format(mag[0], mag[1], mag[2]))

            f.write('\n')


def write_stru(fpath, species, lattice, atoms, orbitals=[]):
    '''
    Generates a ABACUS STRU file.

    Parameters
    ----------
    fpath : str
        Path to the STRU file.
    species : list of dict
        List of atomic species. Each dict contains the following keys:
            - 'symbol' : str
                Element symbol.
            - 'mass' : float (optional)
                Mass of the atomic species. Default to 1.0.
            - 'pp_file' : str
                Name of the pseudopotential file.
            - 'pp_type' : str (optional)
                Type of the pseudopotential file. Default to ''.
    orbitals : list of string (optional)
        List of numerical atomic orbital files. Default to [] (PW).
        If not empty, the length of the list must be the same as that of species.
    lattice : dict
        Parameters to determine the lattice vectors. It contains the following keys:
            - 'latconst' : str
                Lattice constant in Bohr.
            - 'latvec' : list of list of float (optional)
                Lattice vectors in unit of latconst
            - 'latparam' : list of float
                Extra lattice parameters.
    atoms : a two-element list [str, dict]
        The first element is the coordinate type, see
        https://abacus.deepmodeling.com/en/latest/advanced/input_files/stru.html
        for details. The second element is a dict of dict where the first level use
        the element symbols as keys, and the second level contains the following keys:
            - 'mag_each' : float
                Magnetic moment of each atom.
            - 'num' : int
                Number of atoms of this element.
            - 'coord' : list of list of float
                Coordinates of the atoms.
            - 'm' : list of list of float (optional)
                Multiplication factor of the coordinates in relaxation.
            - 'v' : list of list of float (optional)
                Initial velocity of the atoms in relaxation.
            - 'mag' : a two-element list [str, [float, float, float]] (optional)
                Magnetic moment of the atoms. The first element is the coordinate type,
                which can be 'Cartesian' or 'Spherical'. The second element is the
                corresponding [x, y, z] or [r, polar, azimuth].

    '''
    with open(fpath, 'w') as f:
        _write_atomic_species(f, species)
        f.write('\n')
        _write_numerical_orbital(f, orbitals)
        f.write('\n')
        _write_lattice(f, lattice)
        f.write('\n')
        _write_atoms(f, atoms)


def read_stru(fpath):
    '''
    Reads a ABACUS STRU file.

    Returns
    -------
        
    '''
    # TBD
    pass

############################################################
#                       Test
############################################################
import unittest
import os

class _TestSTRUGen(unittest.TestCase):
    def test_write_stru(self):

        fpath = './STRU.test'

        species = [
                {'symbol':'Si', 'mass':28.0   , 'pp_file':'Si.upf', 'pp_type':'upf201' },
                {'symbol':'C'                 , 'pp_file':'C.upf' , 'pp_type':'upf100' },
                {'symbol':'O' , 'mass':32.1233, 'pp_file':'O_ONCV_PBE-1.0.upf' },
                ]

        orbitals = [
                'Si_2s2p1d.orb',
                'C_gga_7au_100Ry.orb',
                'O_gga_7au_100Ry.orb',
                ]

        lattice = {
                'latconst': 22.2,
                'latvec': [
                            [1.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 0.0, 3.0],
                            ],
                'latparam': [1,2],
                }

        atoms = ['Direct',
                 {
                     'Si': {'mag_each': 1.0,
                            'num': 2,
                            'coord': [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                ],
                            'm': [
                                [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0],
                                ],
                            'v': [
                                [0.0, 0.0, -1.0],
                                [0.0, 0.0,  1.0],
                                ],
                            'mag': [
                                ['Cartesian', [0.0, 0.0, 1.0]],
                                ['Spherical', [1.0, 90.0, 180.0]],
                                ],
                            },
                     'C': {'mag_each': 0.0,
                           'num': 1,
                           'coord': [
                               [0.0, 0.0, 0.0],
                               ],
                           },
                     'O': {'mag_each': 1.0,
                           'num': 3,
                           'coord': [
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               ],
                           'm': [
                               [],
                               [1.0, 1.0, 1.0],
                               [],
                               ],
                           'mag': [
                               ['Spherical', [1.0, -72.0, 36.0]],
                               [],
                               ['Cartesian', [1.0, 1.0, -1.0]],
                               ],
                           },
                     }
                 ]

        write_stru(fpath, species, lattice, atoms, orbitals)


############################################################
#                       Main
############################################################
if __name__ == '__main__':
    unittest.main()

