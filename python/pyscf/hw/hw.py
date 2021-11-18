from pyscf import gto, scf
import numpy as np

mol = gto.Mole()

mol.atom = [['Au', (0,0,0)], ['O', (0,0,3)]]
mol.basis = 'def2-svp'
mol.ecp = 'def2-svp'
mol.spin = 1

mol.build()

rhf = scf.UHF(mol)

S = rhf.get_ovlp()

print('size(S,0) = ', np.size(S,0))
print(mol.bas_nctr(0))
print(mol.bas_nprim(0))
print(mol.nao)
print(mol.natm)
print(mol.nbas)
print(mol.nao_nr())
