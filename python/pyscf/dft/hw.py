import numpy as np
from pyscf import dft, gto, scf
import scipy.linalg as sl

class RHF2(scf.hf.RHF):
    __doc__ = scf.hf.RHF.__doc__

    def __init__(self, mol):
        scf.hf.RHF.__init__(self, mol)
        self._rks = dft.RKS(mol)
        self._rks.xc = 'pbe'

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        veff = np.asarray(self._rks.get_veff(dm=dm))
        return veff

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[0] = 2.0
        return mo_occ

h2 = gto.Mole()
h2.atom = '''H 0 0 0; H 0 0 0.74'''
h2.spin=0
h2.basis = 'sto-3g'
h2.build()

hf = scf.RHF(h2)
hf.kernel()
f_hf = hf.get_fock()
print('hf fock = ', f_hf)

ks = dft.RKS(h2)
ks.xc = 'pbe'
ks.kernel()
f_ks = ks.get_fock()
print('ks fock = ', f_ks)

dm = ks.make_rdm1()
ovlp = ks.get_ovlp()
hcore = ks.get_hcore()
JK = ks.get_veff(dm=dm)
fock = hcore+JK
print('fock = ', fock)

e, v = sl.eig(fock, ovlp)
e = e.real
idx = np.argsort(e)
e = e[idx]
v = v[:,idx]
print('e = ', e)

N = np.diag(v.T @ ovlp @ v)
v = v / np.sqrt(N)

mo_coeff = ks.mo_coeff
print('mo_coeff = ', mo_coeff)
print('v = ', v)

print('C S C = ', mo_coeff.T @ ovlp @ mo_coeff)
print('v S v = ', v.T @ ovlp @ v)

P = v[:,0:1] @ v[:,0:1].conj().T * 2



print('dm = ', dm)
print('P = ', P)



exit()

hf2 = RHF2(h2)
hf2.kernel()
dm2 = hf2.make_rdm1()
f_hf2 = hf2.get_fock(dm=dm2)
print('hf2 fock = ', f_hf2)

