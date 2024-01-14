import subprocess

def xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr):
    '''
    Executes ABACUS in a directory.

    Parameters
    ----------
    abacus_path : str
        Path to the ABACUS executable.
    jobdir : str
        Path to the directory where ABACUS is executed.
    nthreads : int
        Number of openmp threads per MPI process.
    nprocs : int
        Number of MPI processes.
    stdout, stderr :
        See the documentation of subprocess.

    '''
    subprocess.run("cd {jobdir}; " \
                   "OMP_NUM_THREADS={nthreads} mpirun -np {nprocs} {abacus_path}" \
                   .format(jobdir=jobdir, nthreads=nthreads, nprocs=nprocs, abacus_path=abacus_path), \
                   shell=True, stdout=stdout, stderr=stderr)


def grep_energy(jobdir, suffix='ABACUS'):
    '''
    Extracts the total energy from the ABACUS output.

    Parameters
    ----------
    jobdir : str
        Path to the directory where ABACUS was executed.
    suffix : str
        Suffix of the ABACUS output directory.

    '''
    result = subprocess.run("grep '!FINAL' {jobdir}/OUT.{suffix}/running_scf.log | awk '{{print $2}}'" \
                            .format(jobdir=jobdir, suffix=suffix),
                            shell=True, capture_output=True, text=True)
    return float(result.stdout)



############################################################
#                       Test
############################################################
import unittest
import shutil
import os

class TestShellTask(unittest.TestCase):
    def test_xabacus_and_grep(self):
        abacus_path = '/home/zuxin/abacus-develop/bin/abacus'
        jobdir = './testfiles/In2/'
        suffix = 'ABACUS'
        nthreads = 2
        nprocs = 4
        stdout = subprocess.DEVNULL
        #stdout = None
        stderr = subprocess.DEVNULL

        shutil.rmtree(jobdir + '/OUT.%s'%(suffix), ignore_errors=True)
        xabacus(abacus_path, jobdir, nthreads, nprocs, stdout, stderr)
        self.assertTrue(os.path.exists(jobdir + '/OUT.%s'%(suffix)))
        self.assertTrue(os.path.exists(jobdir + '/OUT.%s/running_scf.log'%(suffix)))

        energy = grep_energy(jobdir, suffix)
        self.assertTrue(abs(energy - (-2913.0)) < 1.0)
        shutil.rmtree(jobdir + '/OUT.%s'%(suffix), ignore_errors=True)

if __name__ == '__main__':
    unittest.main()

