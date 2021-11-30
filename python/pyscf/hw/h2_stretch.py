from pyscf import gto
from pyscf import scf
import numpy as np
import matplotlib.pyplot as plt

mol_h2 = gto.Mole()
mol_h2.basis = 'sto-3g'

nl = 50
bond_length = np.linspace(0.2, 4, nl)
rhf_tot_energies = np.zeros(nl)
uhf_tot_energies = np.zeros(nl)

for i in range(0,nl):

    mol_h2.atom = [['H',(0, 0, 0)], ['H',(0, 0, bond_length[i])]]
    mol_h2.build()
    
    rhf_h2 = scf.RHF(mol_h2).run()
    rhf_tot_energies[i] = rhf_h2.energy_tot()

    
    uhf_h2 = scf.UHF(mol=mol_h2).newton()
    uhf_h2.max_cycle = 500

    ig = uhf_h2.init_guess_by_minao(mol=mol_h2, breaksym=True)

    #print(ig)
    ig = ig + 0.1*np.random.randn(*np.shape(ig))


    uhf_h2.kernel(dm0=ig)




    uhf_tot_energies[i] = uhf_h2.energy_tot()

plt.plot(bond_length, rhf_tot_energies)
plt.plot(bond_length, uhf_tot_energies)
plt.show()