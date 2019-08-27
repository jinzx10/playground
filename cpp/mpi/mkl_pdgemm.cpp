#include <iostream>
#include <sstream>
#include <fstream>
#include <mkl_pblas.h>
#include <mkl_blacs.h>
#include <mkl_scalapack.h>
#include <mpi.h>
#include "mpiaux.h"

int main(int argc, char** argv) {

	::MPI_Init(nullptr, nullptr);

	int ZERO = 0, ONE = 1;
	int ctxt = 0;
	int id = 0, nprocs = 0;

	::blacs_pinfo(&id, &nprocs);
	::blacs_get(&ZERO, &ZERO, &ctxt);

	// command line input check
	if (argc < 11) {
		if (!id) {
			std::cerr << "Usage: mpirun -np X ./mkl_pdgemm data1.txt M1 N1 data2.txt M2 N2 P Q m n" << std::endl;
		}

		::MPI_Finalize();
		return -1;
	}
	if (!id) {
		std::fstream file("testdata/pdgemm1.txt");

	}


	return 0;
}
