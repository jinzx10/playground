from pyscf import gto
from pyscf import scf


mol_h2o = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = '6-31G')
rhf_h2o = scf.RHF(mol_h2o)
e_h2o = rhf_h2o.kernel()

print(e_h2o)


