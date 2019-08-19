#!/bin/bash

#PBS -l select=3:ncpus=32:mpiprocs=32
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -o /home/zuxin/playground/cpp/mpi/pi.pbs.out

nodes=`cat $PBS_NODEFILE | uniq`
nodes=`echo $nodes | tr ' ' ','`
echo $nodes

bpsh $nodes mkdir -p /scratch/zuxin/pi
bpsh $nodes cp /home/zuxin/playground/cpp/mpi/pi /scratch/zuxin/pi/pi

cd /scratch/zuxin/pi
mpirun -np 96 ./pi 1000000000 > pi.out
cp pi.out /home/zuxin/playground/cpp/mpi/pi.out
bpsh $nodes rm -r /scratch/zuxin/pi

