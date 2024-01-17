import numpy as np

def param2energy(coeff, elem, rcut, sigma, dr, orb_path, abacus_path, jobdirs, \
        coeff_base=None, nthreads=2, nprocs=4, stdout=None, stderr=None, suffix='ABACUS'):
    '''
    Orbital parameters to energy.
    
    Given a set of orbital parameters (spherical Bessel coefficients, cutoff radius, smoothing sigma, etc,
    this function generates an numerical atomic orbital file from those parameters, executes ABACUS in
    specified directories and gets the total energy.
    
    Parameters
    ----------
        coeff : list of list of list of float
            A nested list containing the spherical Bessel coefficients of orbitals in interest.
        elem : str
            Element symbol.
        rcut : float
            Cutoff radius of the orbital.
        sigma : float
            Smoothing parameter.
        dr : float
            Grid spacing.
        orb_path: str
            Path to the orbital file to be generated.
        abacus_path : str
            Path to the ABACUS executable.
        jobdirs : str or list of str
            Directories where ABACUS is executed.
        coeff_base : list of list of list of float
            A nested list containing the spherical Bessel coefficients of 'fixed' orbitals.
        nthreads : int
            Number of threads to be used by ABACUS.
        nprocs : int
            Number of MPI processes to be invoked by ABACUS.
        stdout, stderr :
            See the documentation of subprocess.
    
    '''
    if coeff_base is not None:
        from listmanip import merge
        coeff = merge(coeff_base, coeff, depth=1)

    # generates radial functions
    from radial import build
    chi, r = build(coeff, rcut, dr, sigma, orth=True)

    # writes NAO to an orbital file
    from fileio import write_nao
    write_nao(orb_path, elem, 100, rcut, len(r), dr, chi)

    # executes ABACUS & get energy
    from shelltask import xabacus, grep_energy
    e = 0.0
    jobdirs = [jobdirs] if isinstance(jobdirs, str) else jobdirs
    for jobdir in jobdirs:
        xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr)
        e += grep_energy(jobdir, suffix)

    print('coeff = ', coeff)
    print('total energy = ', e, '\n')
    return e


############################################################
#                       Test
############################################################
import unittest

class TestOrbOpt(unittest.TestCase):
    def test_param2energy(self):
        from fileio import read_param
        param = read_param('./testfiles/ORBITAL_RESULTS.txt')
        dr = 0.01
        orb_path = './testfiles/In_gga_10au_100Ry_3s3p3d2f.orb'
        abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
        jobdirs = './testfiles/In2/'
        e = param2energy(**param, dr=dr, orb_path=orb_path, abacus_path=abacus_path, jobdirs=jobdirs)
        print(e)


############################################################
#                       Main
############################################################
if __name__ == '__main__':
    unittest.main()

#from scipy.optimize import minimize
#
#    from fileio import read_coeff
#    from radbuild import qgen
#    
#    #coeff = read_coeff('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/49_In_DZP/info/7/ORBITAL_RESULTS.txt')
#    #lmax = len(coeff) - 1
#    #nzeta = [len(coeff[l]) for l in range(lmax + 1)]
#    #q = qgen(coeff, rcut)
#    #nq = [len(q[l][izeta]) for l in range(lmax+1) for izeta in range(nzeta[l])]
#    
#    elem = 'In'
#    orbfile = 'In_opt_7au_100Ry_1s1p1d.orb'
#    abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
#    dr = 0.01
#    sigma = 0.1
#    rcut = 7.0
#
#    orbdir = './In/orb/'
#    lens = ['3.00', '3.20', '3.40', '3.60']
#    jobdirs = ['./In/sg15v1.0/' + jobdir + '/' for jobdir in lens ]
#
#    nthreads = 2
#    nprocs = 8
#
#    from subprocess import DEVNULL
#    #stdout=None
#    #stderr=None
#    stdout = DEVNULL
#    stderr = DEVNULL
#
#    ########################################################
#    #           SG15    z_valence = 13 (s+p+d)
#    ########################################################
#    # single-zeta
#    nzeta_sz = [1, 1, 1]
#    lmax_sz = len(nzeta_sz) - 1
#
#    #coeff_sz = [ [ coeff[l][izeta] for izeta in range(nzeta_sz[l]) ] for l in range(lmax_sz+1) ]
#    #q_sz = [ [ q[l][izeta] for izeta in range(nzeta_sz[l]) ] for l in range(lmax_sz+1) ]
#    coeff_sz = read_coeff('./backup/In_sg15v1.0_7au_1s1p1d_22j.txt') # initial guess
#    q_sz = qgen(coeff_sz, rcut)
#    nq_sz = [len(q_sz[l][izeta]) for l in range(lmax_sz+1) for izeta in range(nzeta_sz[l])]
#
#    # sanity check
#    #nzeta_sz = nzeta
#    #lmax_sz = lmax
#    #coeff_sz = coeff
#    #q_sz = q
#    #nq_sz = nq
#    
#    from listmanip import array2list, list2array
#    func_sz = lambda c: param2energy(coeff=array2list(c, lmax_sz, nzeta_sz, nq_sz), q=q_sz, \
#                orbfile=orbfile, elem=elem, rcut=rcut, abacus_path=abacus_path, \
#                coeff_base=None, q_base=None, \
#                dr=dr, sigma=sigma, orbdir=orbdir, jobdirs=jobdirs, \
#                nthreads=nthreads, nprocs=nprocs, stdout=stdout, stderr=stderr)
#    
#    #res = minimize(func_sz, list2array(coeff_sz), method='BFGS', options={'disp': True, 'eps': 1e-3})
#    res = minimize(func_sz, list2array(coeff_sz), method='Nelder-Mead')
#
#    from fileio import write_coeff
#    write_coeff(open('In_sg15v1.0_7au_1s1p1d_22j.txt', 'w'), array2list(res.x, lmax_sz, nzeta_sz, nq_sz), 'In')



    
    
    
