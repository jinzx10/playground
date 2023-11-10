import numpy as np

from scipy.special import spherical_jn
from scipy.integrate import simpson
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize, fmin_bfgs, dual_annealing, basinhopping

import subprocess


'''
Executes ABACUS in a directory.
'''
def xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr):
    subprocess.run("cd {jobdir}; " \
                   "OMP_NUM_THREADS={nthreads} mpirun -np {nprocs} {abacus_path}" \
                   .format(jobdir=jobdir, nthreads=nthreads, nprocs=nprocs, abacus_path=abacus_path), \
                   shell=True, stdout=stdout, stderr=stderr)


'''
Extracts the total energy from the ABACUS output.
'''
def grep_energy(jobdir, suffix='ABACUS'):
    result = subprocess.run("grep '!FINAL' {jobdir}/OUT.{suffix}/running_scf.log | awk '{{print $2}}'" \
                            .format(jobdir=jobdir, suffix=suffix),
                            shell=True, capture_output=True, text=True)
    return float(result.stdout)


'''
Spherical Bessel coefficients to energy.

Given a set of spherical Bessel coefficients, this function generates an ABACUS orbital file
and calls ABACUS to run an SCF calculation to get the energy.

Parameters
----------
    coeff : list of list of list of float
        A nested list containing the coefficients of spherical Bessel functions.
    q : list of list of list of float
        Wave numbers of each spherical Bessel component.
    sigma : float
        Smoothing parameter.
    fname : str
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
def coeff2energy(coeff, q, fname, elem, rcut, \
        dr=0.01, sigma=0.1, orbdir='./', jobdir='./', nthreads=2, nprocs=4):

    print('coeff = ', coeff)

    # generates orbital file
    filegen(orbdir+fname, elem, rcut, coeff, q, dr=dr, sigma=sigma)

    # calls ABACUS to run SCF
    xabacus('/home/zuxin/abacus-develop/bin/abacus', jobdir, \
            nthreads, nprocs, subprocess.DEVNULL, subprocess.DEVNULL)

    # extracts the total energy
    energy = grep_energy(jobdir)
    print('energy = ', result.stdout)

    return energy


'''
Truncates the spherical Bessel coefficients to a smaller number of wave vectors.
'''
def nqfilt(coeff, nq):
    lmax = len(coeff)-1
    nzeta = [len(coeff[l]) for l in range(lmax+1)]
    return [[coeff[l][izeta][:nq] for izeta in range(nzeta[l])] for l in range(lmax+1)]


'''
Plot the radial functions.
'''
def plot_chi(r, chi):
    import matplotlib.pyplot as plt

    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]

    fig, ax = plt.subplots(max(nzeta), lmax+1, squeeze=False)
    for l in range(lmax+1):
        for izeta in range(nzeta[l]):
            ax[izeta,l].plot(r, chi[l][izeta])
            ax[izeta,l].set_xlabel('r')
            ax[izeta,l].set_ylabel('chi')

    plt.show()




########################################################################
#                               main
########################################################################
l1 = [ [[1,2],[2,3]], [[4,5]] ]
l2 = [ [], [[0,1]], [], [[6,7]] ]

#l1a = list2array(l1)
#print(l1a)
#print(array2list(l1a, 1, [2, 1], [2, 2, 2]))

l2a = list2array(l2)
print(l2a)
print(array2list(l2a, 3, [0, 1, 0, 1], [2, 2]))

print('l1 = ', l1)
print('l2 = ', l2)
print(merge(l1, l2))

exit()
#import subprocess
#jobdir='/home/zuxin/tmp/nao/energy_opt/In2/'
#nthreads=3
#nprocs=4
#
#subprocess.run("cd {jobdir}; " \
#               "OMP_NUM_THREADS={nthreads} mpirun -np {nprocs} /home/zuxin/abacus-develop/bin/abacus" \
#               .format(jobdir=jobdir, nthreads=nthreads, nprocs=nprocs), \
#               shell=True, stderr=subprocess.DEVNULL)
#
#exit()

# coefficients from SIAB as initial guess
coeff = read_siab('/home/zuxin/tmp/nao/yike/pd_04/49_In_DZP/info/7/ORBITAL_RESULTS.txt')
lmax = len(coeff)-1
nzeta = [len(coeff[l]) for l in range(lmax+1)]



fname = 'In.orb'
elem = 'In'
rcut = 10
dr = 0.01
sigma = 0.1
nthreads = 2
nprocs = 8

# number of wave vectors for each orbital
#nq_ = 31
nq_ = len(coeff[0][0])
q = [[spherical_jn_root(l, nq_)/rcut for izeta in range(nzeta[l])] for l in range(lmax+1)]

coeff = nqfilt(coeff, nq_)
nq = [nq_] * sum(nzeta)

# working directory
orbdir='/home/zuxin/tmp/nao/energy_opt/In2/'
jobdir='/home/zuxin/tmp/nao/energy_opt/In2/'

c2e = lambda c: coeff2energy(coeff=array2list(c, lmax, nzeta, nq), q=q, \
                fname=fname, elem=elem, rcut=rcut, dr=dr, sigma=sigma, \
                orbdir=orbdir, jobdir=jobdir, nthreads=nthreads, nprocs=nprocs)

######################################
#chi = radgen(coeff, q, rcut, dr, sigma)
#nr = len(chi[0][0])
#r = dr * np.arange(nr)
#plot_chi(r, chi)
#plt.show()
#exit()
######################################

# initial guess
c0 = list2array(coeff)
dim = len(c0)

iiter=1
def disp(c):
    global iiter
    e = c2e(c)
    print('{0}   {1}'.format(iiter, e))
    iiter += 1

# callback function for dual_annealing
def da_disp(x, fmin, context):
    global iiter
    print('{i}   {ctxt}   {val}'.format(i=iiter, ctxt=context, val=fmin))
    iiter += 1
    

#res = basinhopping(c2e, c0)
#res = dual_annealing(c2e, bounds=[(-1,1)]*dim, callback=da_disp, x0=c0, initial_temp=0.1, no_local_search=True, visit=1.5)
#res = minimize(c2e, c0, method='BFGS', callback=disp)
#res = minimize(c2e, c0, method='Nelder-Mead', callback=disp)
res = fmin_bfgs(c2e, c0, epsilon=1e-3, callback=disp)




exit()

chi = radgen(coeff, q, rcut, dr, sigma)
nr = len(chi[0][0])
r = dr * np.arange(nr)
filegen(fname, elem, rcut, coeff, q, dr, sigma)


