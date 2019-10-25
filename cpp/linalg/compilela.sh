#!/bin/bash


g++ zeyu.cpp -o zeyu_2013.out -std=c++11 \
 -m64 -I${MKLROOT}/include \
  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl


#g++ 1.cpp -std=c++11 -lopenblas -llapack -lgfortran
  
