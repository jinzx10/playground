#PBS -N si.scf
#PBS -l nodes=1:ppn=32
#PBS -l mem=4gb
#PBS -l walltime=0:10:00

#PBS -j oe
#PBS -o /home/zuxin/playground/qe/si/si.scf.pbs.out

module load qe/6.4.1

cd /scratch/zuxin
mkdir -p si.scf
cd si.scf

cp $HOME/playground/qe/si/{*.UPF,*.in} .
mpirun -np 32 pw.x -nk 8 -ndiag 4 < si.scf.in > si.scf.out
cp si.scf.out $HOME/playground/qe/si/si.scf.out

cd ..
rm -r si.scf
