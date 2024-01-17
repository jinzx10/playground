# top level directory
top_dir = '/home/zuxin/playground/python/orbgen/'
pseudo_dir = '/home/zuxin/tmp/nao/sg15_oncv_upf_2020-02-06/'
abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
orbital_dir = '/home/zuxin/playground/python/orbgen/scratch/In/sg15v1.2/'
orbital_file = 'In_sg15v1.2_1s1p1d_7au.orb'

import sys
sys.path.append(top_dir)

from inputgen import write_input
from strugen import write_stru
from shelltask import xabacus
from orbopt import param2energy
import pathlib

# dimer bond length
dimer = [2.8, 3.1, 3.4, 3.7]

# sub directory
sub_dir = '/scratch/In/sg15v1.2/lcao/'

for bond_length in dimer:
    job_dir = top_dir + sub_dir + 'dimer_' + str(bond_length) + '/'
    pathlib.Path(job_dir).mkdir(parents=True, exist_ok=True)

    write_input(job_dir,
                orbital_dir=orbital_dir,
                pseudo_dir=pseudo_dir,
                ecutwfc=100,
                scf_nmax=50,
                scf_thr=1e-8,
                basis_type='lcao',
                gamma_only=1,
                )

    species = [
            {'symbol': 'In', 'mass': 1.0, 'pp_file': 'In_ONCV_PBE-1.2.upf'},
            ]

    lattice = {
            'latconst': 20.0,
            'latvec': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                ],
            }

    orbitals = [ orbital_file ]

    atoms = ['Cartesian_angstrom',
             {
                 'In': {
                     'mag_each': 0.0,
                     'num': 2,
                     'coord': [
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, bond_length],
                         ],
                     },
                 }
             ]

    write_stru(job_dir, species, lattice, atoms, orbitals)

from fileio import read_param
from listmanip import *
from scipy.optimize import minimize
import numpy as np
import subprocess

# initial guess
param = read_param('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/49_In_TZDP/info/7/ORBITAL_RESULTS.txt')

# single zeta
param['coeff'] = [[param['coeff'][l][0]] for l in range(3)]

pat = nestpat(param['coeff'])

dr = 0.01
orb_path = orbital_dir + '/' + orbital_file
jobdirs = [top_dir + sub_dir + 'dimer_' + str(bond_length) + '/' for bond_length in dimer]

e_sz = lambda c: param2energy(nest(c.tolist(), pat), param['elem'], param['rcut'], param['sigma'], dr, orb_path, abacus_path, jobdirs, \
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, nthreads=3)

#print(e_sz(  flatten(param['coeff']) ))

coeff_restart =  [[[-0.31530822165098926, -0.27386431875047706, 0.0055556045685686995, 0.1527393585035225, 0.1779567346236594, 0.10631863648513248, 0.04612255519810339, 0.016043510842800233, 0.0008531153518718769, -0.0004902188237545862, -0.0012078393529157344, 0.0010941154091948864, -0.0007613020468791745, 0.0016216113534582442, -0.002054239148768328, 0.0019934127476702217, -0.001256038318689018, 0.0021481362529773024, -0.001929842769463994, 0.002748884679062218, -0.004452795145808963, 0.005050034770699437]], [[0.3586882506479252, 0.2703293856261895, 0.065479271565764, -0.009182499706387095, -0.013509392987716688, -0.050824044121203754, -0.008894722704390682, -0.011796158064662339, 0.004998007894732665, -0.005148592076147837, 0.00663353938527027, -0.005686126769723894, 0.0047686061808073395, -0.010153487160596268, 0.012277524891294252, -0.009680074306268386, 0.0132990695265532, -0.013075163273402735, 0.013713214225960803, -0.01800605284335776, 0.03177629341748907, -0.051408998093488575]], [[-0.08660895684166531, -0.27189870193549337, -0.47945940959170386, -0.6455809596449631, -0.7316446128817462, -0.7204785697494567, -0.6204460674999857, -0.45959716143758056, -0.2850463047250761, -0.14095010321997548, -0.03961487032229949, -3.384150134332431e-05, -0.0017691725502056157, -0.002585194825529793, 7.913966285785385e-05, 0.000395837252925423, -0.0007832346980847168, 0.00010933773507239457, -0.0004928795828703906, 0.0005669366561820446, -0.00026059977997497255, 0.004272058386935795]]] 
res = minimize(e_sz, np.array(flatten(coeff_restart)), method='Nelder-Mead', options={'maxiter': 10000, 'disp': True})


