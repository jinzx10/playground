from pyscf import gto
from pyscf import scf
import numpy as np
import matplotlib.pyplot as plt

mol_h2 = gto.Mole()
mol_h2.basis = 'sto-3g'


bond_length = 2.0

mol_h2.atom = [['H',(0, 0, 0)], ['H',(0, 0, bond_length)]]
mol_h2.build()

#uhf_h2 = scf.UHF(mol_h2).newton()
#uhf_h2.max_cycle = 500
#ig = uhf_h2.init_guess_by_minao(mol=mol_h2, breaksym=True)
#ig = ig + 0.1*np.random.randn(*np.shape(ig))
#uhf_h2.kernel(tol=1e-12,dm0=ig)

#conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol_h2))
ig = np.random.randn(2,2,2)
conv, e, mo_e, mo, mo_occ= scf.hf.kernel(scf.UHF(mol_h2), conv_tol=1e-12, dm0=ig)

print('e = ', e)
print('C = ', mo)

#S = uhf_h2.get_ovlp()
#F = uhf_h2.get_fock()
#C = uhf_h2.mo_coeff
#occ = uhf_h2.mo_occ
#Da = C[0,:,:] @ np.diag(occ[0,:],0) @ C[0,:,:].T
#
#print('S = ', S)
#print('F = ', F)
#print('C = ', C)
#print('occ = ', occ)
#print('Da = ', Da)


