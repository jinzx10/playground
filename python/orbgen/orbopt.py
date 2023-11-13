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
    
    coeff = read_coeff('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/49_In_DZP/info/7/ORBITAL_RESULTS.txt')
    lmax = len(coeff) - 1
    nzeta = [len(coeff[l]) for l in range(lmax + 1)]
    nq_ = len(coeff[0][0])
    rcut = 7.0
    q = [[ikebe(l, nq_)/rcut for izeta in range(nzeta[l])] for l in range(lmax+1)]
    nq = [len(q[l][izeta]) for l in range(lmax+1) for izeta in range(nzeta[l])]
    
    elem = 'In'
    orbfile = 'In_opt_7au_100Ry_1s1p1d.orb'
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
    # single-zeta
    nzeta_sz = [1, 1, 1]
    lmax_sz = len(nzeta_sz) - 1

    coeff_sz = [ [ coeff[l][izeta] for izeta in range(nzeta_sz[l]) ] for l in range(lmax_sz+1) ]
    q_sz = [ [ q[l][izeta] for izeta in range(nzeta_sz[l]) ] for l in range(lmax_sz+1) ]
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
    
    #print(func_sz(list2array(coeff_sz)))
    #exit()

    res = fmin_bfgs(func_sz, list2array(coeff_sz), epsilon=1e-3)





    exit()
    
    
    
    
    coeff_extra = [ [ coeff[l][izeta] for izeta in range(nzeta_base[l], nzeta[l]) ] for l in range(lmax+1) ]
    q_extra = [ [ q[l][izeta] for izeta in range(nzeta_base[l], nzeta[l]) ] for l in range(lmax+1) ]
    
    orbdir = './In2/'
    
    
    e = extra_coeff2target(coeff_base, q_base, coeff_extra, q_extra, 'In.orb', 'In', rcut, '/home/zuxin/abacus-develop/bin/abacus', orbdir=orbdir, jobdirs=jobdirs)
    print('e = ', e)
    
    exit()
    #e = coeff2energy(coeff, q, 'In.orb', 'In', rcut, '/home/zuxin/abacus-develop/bin/abacus', orbdir=orbdir, jobdir=jobdir)
    #print('e = ', e)
    #exit()
    
    e = coeff2energy(coeff_base, q_base, 'In.orb', 'In', rcut, '/home/zuxin/abacus-develop/bin/abacus', orbdir=orbdir, jobdir=jobdir)
    print('e = ', e)
    
    
    e = extra_coeff2energy(coeff_base, q_base, coeff_extra, q_extra, 'In.orb', 'In', rcut, '/home/zuxin/abacus-develop/bin/abacus', orbdir=orbdir, jobdir=jobdir)
    print('e = ', e)
