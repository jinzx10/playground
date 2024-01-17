# top level directory
top_dir = '/home/zuxin/playground/python/orbgen/'
abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
pseudo_dir = '/home/zuxin/tmp/nao/sg15_oncv_upf_2020-02-06/'

symbol = 'Hf'
pp_type = 'sg15v1.0'
pp_file = 'Hf_ONCV_PBE-1.0.upf'

import sys
sys.path.append(top_dir)

from inputgen import write_input
from strugen import write_stru
from shelltask import xabacus
import pathlib

# dimer bond length
dimer = [2.5, 2.8, 3.1, 3.4, 3.7, 4.0]

# sub directory
sub_dir = '/scratch/{symbol}/{pp_type}/pw/'%(symbol=symbol, pp_type=pp_type)

for bond_length in dimer:
    job_dir = top_dir + sub_dir + 'dimer_' + str(bond_length) + '/'
    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)

    write_input(job_dir,
                pseudo_dir=pseudo_dir,
                ecutwfc=100,
                scf_nmax=50,
                scf_thr=1e-8,
                basis_type='pw',
                gamma_only=1,
                )

    species = [
            {'symbol': symbol, 'mass': 1.0, 'pp_file': pp_file},
            ]

    lattice = {
            'latconst': 20.0,
            'latvec': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                ],
            }

    atoms = ['Cartesian_angstrom',
             {
                 symbol : {
                     'mag_each': 0.0,
                     'num': 2,
                     'coord': [
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, bond_length],
                         ],
                     },
                 }
             ]

    write_stru(job_dir, species, lattice, atoms)

    xabacus(abacus_path, job_dir, 1, 4, None, None)




