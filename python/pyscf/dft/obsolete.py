from pyscf import gto, dft
import numpy as np
import matplotlib.pyplot as plt

class RKS2(dft.rks.RKS):

    __doc__ = dft.rks.RKS.__doc__

    def __init__(self, mol, xc, mu, smearing=None):
        self.mu = mu
        self.smearing = smearing
        dft.rks.RKS.__init__(self, mol, xc)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        if self.smearing:
            for n,e in enumerate(mo_energy):
                mo_occ[n] = 2./(numpy.exp((e-self.mu)/self.smearing)+1)
        else:
            mo_occ[mo_energy<=self.mu] = 2.
        nmo = mo_energy.size
        nocc = int(numpy.sum(mo_occ) // 2)
        if self.verbose >= logger.INFO and nocc < nmo:
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, 'HOMO %.15g == LUMO %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])
            else:
                logger.info(self, '  nelec = %d', nocc*2)
                logger.info(self, '  HOMO = %.15g  LUMO = %.15g',
                            mo_energy[nocc-1], mo_energy[nocc])

        if self.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=nmo)
            logger.debug(self, '  mo_energy =\n%s', mo_energy)
            numpy.set_printoptions(threshold=1000)
        return mo_occ
    def get_veff(self, m, dm):
        veff = np.zeros_like(dm)

        # build an artificial object
        mol = gto.M()
        mol.atom = '''Co 0 0 0'''
        mol.spin = 0
        mol.nelectron=26
        mol.basis = 'def2-svp'
        mol.build()
        mf = dft.RKS(mol)

        '''
        # val+virt part
        dm_Co_vv = dm[0:22,0:22]

        # add core in LO basis
        dm_Co_tot_lo = 

        # convert to AO
        dm_Co_ao = 

        # get ordinary RKS
        veff_Co = mf.get_veff(dm_Co_ao)

        # convert to LO
        veff_Co = 

        # get val+virt block
        veff_Co_vv = 

        # paste to the Co block in veff
        veff[] = veff_Co_vv
        '''

        return veff


mol = gto.M()
mol.atom = ''' Co 0 0 0 '''
mol.spin = 0
#mol.charge = 1
mol.nelectron=28
mol.basis = 'def2-svp'
mol.build()

mf = dft.RKS(mol)
hcore = mf.get_hcore()
ovlp = mf.get_ovlp()
print('hcore.shape = ', hcore.shape)
mf.kernel()
dm = mf.make_rdm1()
print('dm.shape = ', dm.shape)
print('trace(dm) = ', np.trace(dm@ovlp))
#print('trace(dm) = ', np.trace(dm[0]))
#print('trace(dm) = ', np.trace(dm[1]))

plt.imshow(np.abs(dm), extent=[0,1,0,1])
plt.show()

JK = mf.get_veff()
print('JK.shape = ', JK.shape)

dm2 = dm.copy()

idx_core = [0,1,2,5,6,7,8,9,10]
for i in idx_core:
    dm2[i,:] = 0
    dm2[:,i] = 0
    dm2[i,i] = 1
JK2 = mf.get_veff(dm=dm2)
print('trace(dm2) = ', np.trace(dm2@ovlp))
print('JK diff = ', np.linalg.norm(JK2-JK))

dm3 = np.zeros_like(dm2)




