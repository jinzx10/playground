#include <mpi.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <armadillo>
#include <string>
#include "../utility/mpi_helper.h"
#include "helper.h"
#include "../utility/widgets.h"


int main(int, char** argv) {

	int id, nprocs;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	arma::mat A, B, C;
	int rA, cA, cB;

	if (id == 0) {
		readargs(argv, rA, cA, cB);
		A.randu(rA, cA);
		B.randu(cA, cB);
		C.set_size(rA, cB);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (id == 0) {
		std::cout << "ready" << std::endl;
	}
	
	pmatmul(A, B, C);
	
	if (id == 0) {
		C.print();
		std::cout << std::endl;
		(A*B).print();
	}

	MPI_Finalize();

}
