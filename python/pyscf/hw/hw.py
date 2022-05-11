from pyscf import gto, scf
import numpy as np

mol = gto.Mole()

mol.atom = [['Au', (0,0,0)], ['O', (0,0,3)]]
#mol.basis = 'def2-svp'
mol.basis = {'Au':'def2-svp', 'O':'sto-3g'}
mol.ecp = {'Au':'def2-svp'}

print('mol.nelectron = ', mol.nelectron)

mol.spin = mol.nelectron%2


#mol.spin = 1
mol.build()

print('mol.basis = ', mol.basis)
print(type(mol.basis))
print('mol.ecp = ', mol.ecp)
print(type(mol.ecp))

rhf = scf.UHF(mol)

S = rhf.get_ovlp()

print('size(S,0) = ', np.size(S,0))
print(mol.nao)
print(mol.nao_nr())
