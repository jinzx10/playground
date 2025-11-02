#!/bin/bash

rm -f build/CMakeCache.txt
cmake -B build
cmake --build build

mpirun -np 4 ./build/hw
