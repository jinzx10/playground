import numpy as np

from scipy.optimize import minimize, fmin_bfgs, dual_annealing, basinhopping




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
def plot_chi(r, chi, chi2=None, label=None, label2=None):
    import matplotlib.pyplot as plt

    lmax = len(chi)-1
    nzeta = [len(chi[l]) for l in range(lmax+1)]

    fig, ax = plt.subplots(max(nzeta), lmax+1, squeeze=False)
    for l in range(lmax+1):
        for izeta in range(nzeta[l]):
            ax[izeta,l].plot(r, chi[l][izeta], label=label)
            if chi2 is not None:
                ax[izeta,l].plot(r, chi2[l][izeta], label=label2)
            ax[izeta,l].set_xlabel('r')
            ax[izeta,l].set_ylabel('chi')
            ax[izeta,l].legend()

    plt.show()




########################################################################
#                               main
########################################################################
from radbuild import j2rad, qgen
from jnroot import ikebe
from fileio import read_coeff
from listmanip import merge

rcut = 7
coeff_ref = read_coeff('/home/zuxin/tmp/nao/v2.0/SG15-Version1p0__AllOrbitals-Version2p0/49_In_DZP/info/7/ORBITAL_RESULTS.txt')

# optimized minimal
coeff_min_opt  =  [[[-0.22319273382936658, -0.1935692770353136, 0.011898484474316167, 0.11687347789973701, 0.12424724776710969, 0.082826751969036, 0.039426527485066025, 0.009350748438199118, -0.002711813172651745, -0.004928288984114024, -0.003067808241357446, -0.0010580361641553372, -0.0011439640018363016, -0.0002709781545847151, -0.001356846431694946, 0.0002785320787625634, -0.0012951057606575107, 0.000823205142996889, -0.001533729202231069, 0.0013267383747682941, -0.002243739821384359, 0.0028827686740655204]], [[0.31489881082247795, 0.23032926246318386, 0.07452029206069405, -0.011082935875728539, -0.040694592814021574, -0.03391193894336489, -0.019691111499707623, -0.005795739729858165, -0.0010559715835145814, 0.00021013426442062743, -0.0010804442170462983, -0.0018444567909232613, -0.0011416021931900022, -0.0013896905727685176, 0.00010567313703063635, -0.0013384021875936208, 0.0005445377048263758, -0.0017427974121259394, 0.0013164127584227412, -0.0024294995719134653, 0.0026845260388395496, -0.00962980040832163]], [[-0.07724511065901557, -0.24018330937918517, -0.41667348834272955, -0.5612345764271982, -0.6432834945639484, -0.6561132968855918, -0.5986469171723564, -0.49224502020309474, -0.36042136532934477, -0.23269166151240536, -0.12882682275958157, -0.058179644745669806, -0.019879759662448166, -0.0032359945413300128, 3.9697296477234794e-05, 0.0006239628995839741, -0.0009135870763134326, 0.00031130486079140147, -0.00047062473024814613, 0.0012295324888643904, -0.00012199965090200087, -0.003022552040389695]]]

# optimized s2
coeff_s2_opt =  [[[-1.0015057670927148, 1.070592032949412, 0.25800982539745304, -0.9949142819096615, -1.0801318639885973, -0.6040751569855636, 0.15102266517508775, 0.41617198194308785, 0.6468826306910692, 0.45230181267530184, 0.4523933582730802, 0.1621021220171285, 0.2486521588148511, 0.016149099108352633, 0.16781890114635395, -0.0516393703226241, 0.142803132169651, -0.07827873350920239, 0.13817720827070487, -0.11869133395516576, 0.19651068246713685, -0.2478793374867025]]]

coeff_opt = merge(coeff_min_opt, coeff_s2_opt)

# optimized p2
coeff_p2_opt =  [[], [[1.5495871001609018, -1.7494958207723248, -0.2447651272030374, -0.24117121001938188, 0.2256338466895182, -0.08143314508381676, 0.14268490377431534, -0.23792631938942313, 0.0046811262478916186, -0.2523279599568605, 0.026571372733734923, -0.18364619426964574, 0.05660782095003873, -0.17383987557442881, 0.06217441978794814, -0.17539928787760842, 0.07484416691656243, -0.1746027575467987, 0.12545567705787156, -0.22468533990850564, 0.22564833046904137, -0.8837360087769349]]]


coeff_opt = merge(coeff_opt, coeff_p2_opt)

# old
coeff_old = [[coeff_ref[0][0], coeff_ref[0][1]], [coeff_ref[1][0], coeff_ref[1][1]], [coeff_ref[2][0]]]


lmax = len(coeff_old)-1
nzeta = [len(coeff_old[l]) for l in range(lmax+1)]
nq_ = len(coeff_ref[0][0])
q = qgen(nzeta, nq_, rcut)
dr = 0.01
nr = int(rcut/dr) + 1
rcut = 7
r = dr * np.arange(nr)  

chi_opt = j2rad(coeff_opt, q, rcut)
chi_old = j2rad(coeff_old, q, rcut)

plot_chi(r, chi_old, chi_opt, label='old', label2='opt')
exit()



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


