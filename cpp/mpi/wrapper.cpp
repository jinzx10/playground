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
	Stopwatch sw;

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
		sw.run();
	}
	
	pmatmul(A, B, C);
	
	if (id == 0) {
		sw.report("pdgemm");
		sw.reset();
		sw.run();
		arma::mat D = A*B;
		sw.report("single-core");
		arma::mat err = D - C;
		std::cout << "error = " << arma::accu(err % err) << std::endl;
#ifdef PRINT
		A.print();
		std::cout << std::endl;
		B.print();
		std::cout << std::endl;
		C.print();
		std::cout << std::endl;
		D.print();
		std::cout << std::endl;
#endif
	}

	MPI_Finalize();

}
