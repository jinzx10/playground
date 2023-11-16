import numpy as np
from scipy.optimize import fmin_bfgs

'''
Spherical Bessel coefficients to energy.

Given a set of spherical Bessel wave numbers and coefficients, this function generates
an orbital file, executes ABACUS in a directory and gets the total energy.

Parameters
----------
    coeff : list of list of list of float
        A nested list containing the coefficients of spherical Bessel functions.
    q : list of list of list of float
        Wave numbers of each spherical Bessel component.
    sigma : float
        Smoothing parameter.
    orbfile : str
        Name of the orbital file to be generated.
    elem : str
        Element symbol.
    rcut : int or float
        Cutoff radius of the orbital.
    jobdir : str
        Directory to run the SCF calculation.
    nthreads : int
        Number of threads to be used in the SCF calculation.
    nprocs : int
        Number of MPI processes to be used in the SCF calculation.
'''
def coeff2energy(coeff, q, orbfile, elem, rcut, abacus_path, coeff_base=None, q_base=None, \
        dr=0.01, sigma=0.1, orbdir='./', jobdirs=['./'], nthreads=2, nprocs=4, stdout=None, stderr=None):
    print('coeff = ', coeff)
    if coeff_base is not None:
        from listmanip import merge
        coeff = merge(coeff_base, coeff)
        q = merge(q_base, q)

    # generates radial functions
    from radbuild import j2rad
    chi = j2rad(coeff, q, rcut, dr, sigma)

    # writes to an orbital file
    from fileio import write_orbfile
    write_orbfile(orbdir + '/' + orbfile, elem, rcut, chi, dr=dr)

    # calls ABACUS to run SCF & get energy
    from shelltask import xabacus, grep_energy
    e = 0.0
    for jobdir in jobdirs:
        xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr)
        e += grep_energy(jobdir)
    print('total energy = ', e, '\n')
    return e


############################################################
#                       Testing
############################################################

def test_coeff2energy():
    pass


if __name__ == '__main__':

    from fileio import read_coeff
    from jnroot import ikebe
    
    coeff_ref = read_coeff('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/49_In_DZP/info/7/ORBITAL_RESULTS.txt')

    # minimal basis
    coeff_base =  [[[-0.22319273382936658, -0.1935692770353136, 0.011898484474316167, 0.11687347789973701, 0.12424724776710969, 0.082826751969036, 0.039426527485066025, 0.009350748438199118, -0.002711813172651745, -0.004928288984114024, -0.003067808241357446, -0.0010580361641553372, -0.0011439640018363016, -0.0002709781545847151, -0.001356846431694946, 0.0002785320787625634, -0.0012951057606575107, 0.000823205142996889, -0.001533729202231069, 0.0013267383747682941, -0.002243739821384359, 0.0028827686740655204]], [[0.31489881082247795, 0.23032926246318386, 0.07452029206069405, -0.011082935875728539, -0.040694592814021574, -0.03391193894336489, -0.019691111499707623, -0.005795739729858165, -0.0010559715835145814, 0.00021013426442062743, -0.0010804442170462983, -0.0018444567909232613, -0.0011416021931900022, -0.0013896905727685176, 0.00010567313703063635, -0.0013384021875936208, 0.0005445377048263758, -0.0017427974121259394, 0.0013164127584227412, -0.0024294995719134653, 0.0026845260388395496, -0.00962980040832163]], [[-0.07724511065901557, -0.24018330937918517, -0.41667348834272955, -0.5612345764271982, -0.6432834945639484, -0.6561132968855918, -0.5986469171723564, -0.49224502020309474, -0.36042136532934477, -0.23269166151240536, -0.12882682275958157, -0.058179644745669806, -0.019879759662448166, -0.0032359945413300128, 3.9697296477234794e-05, 0.0006239628995839741, -0.0009135870763134326, 0.00031130486079140147, -0.00047062473024814613, 0.0012295324888643904, -0.00012199965090200087, -0.003022552040389695]]]

    nq_ = len(coeff_ref[0][0])
    rcut = 7.0
    lmax_base = len(coeff_base) - 1
    nzeta_base = [len(coeff_base[l]) for l in range(lmax_base + 1)]
    q_base = [[ikebe(l, nq_)/rcut for izeta in range(nzeta_base[l])] for l in range(lmax_base+1)]

    elem = 'In'
    orbfile = 'In_opt_7au_100Ry_2s1p1d.orb'
    abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
    dr = 0.01
    sigma = 0.1

    #e = coeff2energy(coeff, q, orbfile, elem, rcut, abacus_path, coeff_base=None, q_base=None, \
    #    dr=0.01, sigma=0.1, orbdir='./In/orb/', jobdir='./In/sg15v1.0/3.00/', \
    #    nthreads=2, nprocs=4, stdout=None, stderr=None)
    #print(e)
    #exit()


    orbdir = './In/orb/'
    lens = ['3.00', '3.20', '3.40', '3.60']
    jobdirs = ['./In/sg15v1.0/' + jobdir + '/' for jobdir in lens ]

    nthreads = 2
    nprocs = 8

    from subprocess import DEVNULL
    #stdout=None
    #stderr=None
    stdout = DEVNULL
    stderr = DEVNULL

    ########################################################
    #           SG15    z_valence = 13 (s+p+d)
    ########################################################
    # minimal basis plus extra s
    nzeta = [2, 1, 1]
    lmax = len(nzeta) - 1

    # extra s orbital
    coeff = [ [ coeff_ref[0][1] ] ]
    q = [ [ ikebe(0, nq_)/rcut ] ]

    from listmanip import array2list, list2array
    func = lambda c: coeff2energy(coeff=array2list(c, 0, [1], [nq_]), q=q, \
                orbfile=orbfile, elem=elem, rcut=rcut, abacus_path=abacus_path, \
                coeff_base=coeff_base, q_base=q_base, \
                dr=dr, sigma=sigma, orbdir=orbdir, jobdirs=jobdirs, \
                nthreads=nthreads, nprocs=nprocs, stdout=stdout, stderr=stderr)

    #print(func(list2array(coeff)))
    
    res = fmin_bfgs(func, list2array(coeff), epsilon=1e-3)




