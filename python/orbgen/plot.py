import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def plot_chi(r, chi, r_ref=None, chi_ref=None, label=None, label_ref=None):
    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]

    fig, ax = plt.subplots(max(nzeta), lmax+1, squeeze=False, figsize=(8*(lmax+1), 6*max(nzeta)), layout='tight')
    for l in range(lmax+1):
        for izeta in range(nzeta[l]):
            ax[izeta,l].plot(r, chi[l][izeta], label=label)
            if chi_ref is not None:
                ax[izeta,l].plot(r_ref, chi_ref[l][izeta], label=label_ref)

            ax[izeta,l].set_xlim([0, r[-1] if r_ref is None else max(r[-1], r_ref[-1])])
            ax[izeta,l].axhline(0,0, color='k', ls='--')
            ax[izeta,l].set_xlabel('$r$', fontsize=20)
            ax[izeta,l].set_ylabel('$\chi(r)$', fontsize=20)
            ax[izeta,l].set_title('$l=$%i ~~~~$\zeta=$%i'%(l,izeta), fontsize=24)
            ax[izeta,l].legend(fontsize=20)
            ax[izeta,l].xaxis.set_tick_params(labelsize=16)
            ax[izeta,l].yaxis.set_tick_params(labelsize=16)

    plt.show()


def plot_coeff(coeff, rcut, q=None, coeff_ref=None, rcut_ref=None, q_ref=None, dr=0.01, sigma=0.1, label=None, label_ref=None):
    from radbuild import qgen, j2rad
    if q is None:
        q = qgen(coeff, rcut)
    chi, r = j2rad(coeff, q, rcut, dr, sigma)

    chi_ref=None
    r_ref=None
    if coeff_ref is not None:
        chi_ref, r_ref = j2rad(coeff_ref, q_ref if q_ref is not None else qgen(coeff_ref, rcut), rcut_ref, dr, sigma)

    plot_chi(r, chi, r_ref, chi_ref, label, label_ref)


if __name__ == '__main__':
    from fileio import read_coeff
    from radbuild import qgen
    #coeff = read_coeff('./testfiles/In_sg15v1.0_7au_1s1p1d.coeff.txt')
    coeff =  [[[-0.22830906019015523, -0.19639911565887214, 0.013496270369343713, 0.11630106344419208, 0.12591180273560976, 0.08300525575306442, 0.041006913800616565, 0.009411812063463396, -0.0029713427141206446, -0.0052986929229751935, -0.0028219228893889376, -0.0011585638317863542, -0.0010122622734065213, -0.0002439959284480926, -0.0023637460526181736, 0.00023270119375999186, -0.0012538716234406404, 0.000559625684572353, -0.002119508127796865, 0.0010279135836585443, -0.002328015194119043, 0.0026872427160474456]], [[0.3404956432682833, 0.2501406167739807, 0.07621272805999735, -0.01036973036438664, -0.045739752776836354, -0.03499561906299575, -0.023065123804037857, -0.005690005518908508, -0.0013816329730549897, 0.00019185931049971514, -0.0006910296218480578, -0.0013074294302497223, -0.0009602179300326917, -0.0014364304118816783, 0.0001566323635438081, -0.0012691400753611073, 0.0001282369623962675, -0.0019287956273814372, 0.0011927935070970615, -0.002735150766869066, 0.003593817125844493, -0.0011417012703874765]], [[-0.07673919579119193, -0.23796921499792645, -0.41608366368980887, -0.5603599393350478, -0.6433859012269438, -0.6550077236173419, -0.5987917228197877, -0.4920570815634213, -0.3612808893734615, -0.2334056358882745, -0.12953728221344896, -0.05896942586309026, -0.01995918419465549, -0.003645314824302202, 3.140135445530638e-05, 0.0009100050463258982, -0.0011954703411422808, 0.0004428799404913549, -0.00045687161636482937, 0.0011100563360295409, -0.00011226976613397236, -0.000556328324606457]]]

    coeff_ref = read_coeff('/home/zuxin/tmp/nao/v1.0/Orb_SG15_DZP_E100_Full_v1.0/In/info/8/ORBITAL_RESULTS.txt')
    coeff_ref = [[coeff_ref[0][0]], [coeff_ref[1][0]], [coeff_ref[2][0]]]
    plot_coeff(coeff, 7.0, dr=0.01, sigma=0.1, label='sg15v1.0', coeff_ref=coeff_ref, q=qgen(coeff, 7.0), label_ref='v1.0', q_ref=qgen(coeff_ref, 8.0), rcut_ref = 8.0)

    coeff_ref = read_coeff('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/49_In_DZP/info/7/ORBITAL_RESULTS.txt')
    plot_coeff(coeff, 7.0, dr=0.01, sigma=0.1, label='sg15v1.0', coeff_ref=coeff_ref, q=qgen(coeff, 7.0), label_ref='v2.0', q_ref=qgen(coeff_ref, 7.0), rcut_ref = 7.0)

