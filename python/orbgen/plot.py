import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def plot_chi(r, chi, chi_ref=None, label=None, label2=None):
    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]

    fig, ax = plt.subplots(max(nzeta), lmax+1, squeeze=False, figsize=(8*(lmax+1), 6*max(nzeta)), layout='tight')
    for l in range(lmax+1):
        for izeta in range(nzeta[l]):
            ax[izeta,l].plot(r, chi[l][izeta], label=label)
            if chi_ref is not None:
                ax[izeta,l].plot(r, chi_ref[l][izeta], label=label2)

            ax[izeta,l].set_xlim([0, r[-1]])
            ax[izeta,l].axhline(0,0, color='k', ls='--')
            ax[izeta,l].set_xlabel('$r$', fontsize=20)
            ax[izeta,l].set_ylabel('$\chi(r)$', fontsize=20)
            ax[izeta,l].set_title('$l=$%i ~~~~$\zeta=$%i'%(l,izeta), fontsize=24)
            ax[izeta,l].legend(fontsize=20)
            ax[izeta,l].xaxis.set_tick_params(labelsize=16)
            ax[izeta,l].yaxis.set_tick_params(labelsize=16)

    plt.show()


def plot_coeff(coeff, rcut, q=None, coeff_ref=None, q_ref=None, dr=0.01, sigma=0.1, label=None, label2=None):
    from radbuild import qgen, j2rad
    if q is None:
        q = qgen(coeff, rcut)
    chi, r = j2rad(coeff, q, rcut, dr, sigma)

    chi_ref=None
    if coeff_ref is not None and q_ref is None:
        q_ref = qgen(coeff_ref, rcut)
        chi_ref, _ = j2rad(coeff_ref, q_ref, rcut, dr, sigma)

    plot_chi(r, chi, chi_ref, label, label2)


if __name__ == '__main__':
    from fileio import read_coeff
    coeff = read_coeff('./testfiles/In_sg15v1.0_7au_1s1p1d.coeff.txt')
    #coeff = read_coeff('./testfiles/ORBITAL_RESULTS.txt')
    plot_coeff(coeff, 7.0, dr=0.01, sigma=0.1, label='sg15v1.0')
