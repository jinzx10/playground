import subprocess

def xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr):
    '''
    Executes ABACUS in a directory.

    '''
    subprocess.run("cd {jobdir}; " \
                   "OMP_NUM_THREADS={nthreads} mpirun -np {nprocs} {abacus_path}" \
                   .format(jobdir=jobdir, nthreads=nthreads, nprocs=nprocs, abacus_path=abacus_path), \
                   shell=True, stdout=stdout, stderr=stderr)


def grep_energy(jobdir, suffix='ABACUS'):
    '''
    Extracts the total energy from the ABACUS output.

    '''
    result = subprocess.run("grep '!FINAL' {jobdir}/OUT.{suffix}/running_scf.log | awk '{{print $2}}'" \
                            .format(jobdir=jobdir, suffix=suffix),
                            shell=True, capture_output=True, text=True)
    return float(result.stdout)



############################################################
#                       Testing
############################################################
def test_xabacus():
    abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
    jobdir = './testfiles/In2'
    nthreads = 2
    nprocs = 4
    #stdout = subprocess.DEVNULL
    stdout = None
    stderr = subprocess.DEVNULL
    xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr)

if __name__ == '__main__':
    test_xabacus()

