#!/bin/bash

#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -a 1535
#PBS -j oe
#PBS -o /home/zuxin/pg.pbs.out

cd /home/zuxin/playground/cpp/hw

./hw
