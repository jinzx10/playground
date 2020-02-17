#!/bin/bash

g++ zheevd.cpp -o zheevd \
	-larmadillo -L${MKLROOT}/lib/intel64 \
	-Wl,--no-as-needed -lmkl_intel_lp64 \
	-lmkl_sequential -lmkl_core -lpthread -lm -ldl
