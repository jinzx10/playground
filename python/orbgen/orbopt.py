import numpy as np

def coeff2energy(coeff, orbfile, elem, rcut, abacus_path, coeff_base=None, q=None, q_base=None, \
        dr=0.01, sigma=0.1, orbdir='./', jobdirs=['./'], nthreads=2, nprocs=4, stdout=None, stderr=None):
    '''
    Orbital parameters to energy.
    
    Given a set of orbital parameters (spherical Bessel coefficients, cutoff radius, smoothing sigma, etc,
    this function generates an numerical atomic orbital file from those parameters, executes ABACUS in
    specified directories and gets the total energy.
    
    Parameters
    ----------
        coeff : list of list of list of float
            A nested list containing the spherical Bessel coefficients of orbitals in interest.
        orbfile : str
            Name of the orbital file to be generated. (Not include the directory)
        elem : str
            Element symbol.
        rcut : float
            Cutoff radius of the orbital.
        abacus_path : str
            Path to the ABACUS executable.
        coeff_base : list of list of list of float
            A nested list containing the spherical Bessel coefficients of 'fixed' orbitals.
        q : list of list of list of float
            Wave numbers of each spherical Bessel function in coeff.
            If None, will be generated from rcut & the size of coeff.
        q_base : list of list of list of float
            Wave numbers of each spherical Bessel function in coeff_base.
            If None, will be generated from rcut & the size of coeff_base.
        dr : float
            Grid spacing.
        sigma : float
            Smoothing parameter.
        orbdir : str
            Directory to write the orbital file.
        jobdirs : list of str
            Directories to where ABACUS is executed.
        nthreads : int
            Number of threads to be used by ABACUS.
        nprocs : int
            Number of MPI processes to be invoked by ABACUS.
        stdout, stderr :
            See the documentation of subprocess.
    
    '''
    print('coeff = ', coeff)

    if coeff_base is not None:
        from listmanip import merge
        coeff = merge(coeff_base, coeff, depth=1)

    # generates radial functions
    from radial import build
    chi, r = build(coeff, rcut, dr, sigma, q=None, orth=False)

    # writes to an orbital file
    from fileio import write_nao
    write_nao(orbdir + '/' + orbfile, elem, 100, rcut, len(r), dr, chi)

    # calls ABACUS to run SCF & get energy
    from shelltask import xabacus, grep_energy
    e = 0.0
    for jobdir in jobdirs:
        xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr)
        e += grep_energy(jobdir)

    print('total energy = ', e, '\n')
    return e


def inputgen():
    '''
    Generates ABACUS input files for orbital optimizations.
    '''
    pass

############################################################
#                       Test
############################################################
import unittest

def TestOrbOpt(unittest.TestCase):
    pass


############################################################
#                       Main
############################################################
from scipy.optimize import minimize

if __name__ == '__main__':

    from fileio import read_coeff
    from radbuild import qgen
    
    #coeff = read_coeff('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/49_In_DZP/info/7/ORBITAL_RESULTS.txt')
    #lmax = len(coeff) - 1
    #nzeta = [len(coeff[l]) for l in range(lmax + 1)]
    #q = qgen(coeff, rcut)
    #nq = [len(q[l][izeta]) for l in range(lmax+1) for izeta in range(nzeta[l])]
    
    elem = 'In'
    orbfile = 'In_opt_7au_100Ry_1s1p1d.orb'
    abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
    dr = 0.01
    sigma = 0.1
    rcut = 7.0

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
    # single-zeta
    nzeta_sz = [1, 1, 1]
    lmax_sz = len(nzeta_sz) - 1

    #coeff_sz = [ [ coeff[l][izeta] for izeta in range(nzeta_sz[l]) ] for l in range(lmax_sz+1) ]
    #q_sz = [ [ q[l][izeta] for izeta in range(nzeta_sz[l]) ] for l in range(lmax_sz+1) ]
    coeff_sz = read_coeff('./backup/In_sg15v1.0_7au_1s1p1d_22j.txt') # initial guess
    q_sz = qgen(coeff_sz, rcut)
    nq_sz = [len(q_sz[l][izeta]) for l in range(lmax_sz+1) for izeta in range(nzeta_sz[l])]

    # sanity check
    #nzeta_sz = nzeta
    #lmax_sz = lmax
    #coeff_sz = coeff
    #q_sz = q
    #nq_sz = nq
    
    from listmanip import array2list, list2array
    func_sz = lambda c: coeff2energy(coeff=array2list(c, lmax_sz, nzeta_sz, nq_sz), q=q_sz, \
                orbfile=orbfile, elem=elem, rcut=rcut, abacus_path=abacus_path, \
                coeff_base=None, q_base=None, \
                dr=dr, sigma=sigma, orbdir=orbdir, jobdirs=jobdirs, \
                nthreads=nthreads, nprocs=nprocs, stdout=stdout, stderr=stderr)
    
    #res = minimize(func_sz, list2array(coeff_sz), method='BFGS', options={'disp': True, 'eps': 1e-3})
    res = minimize(func_sz, list2array(coeff_sz), method='Nelder-Mead')

    from fileio import write_coeff
    write_coeff(open('In_sg15v1.0_7au_1s1p1d_22j.txt', 'w'), array2list(res.x, lmax_sz, nzeta_sz, nq_sz), 'In')



    
    
    
