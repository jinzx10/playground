#include <mpi.h>
#include <armadillo>
#include "../utility/mpi_helper.h"
#include "../utility/widgets.h"

int main(int, char**argv) {

	int id, nprocs;
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int sz;
	Stopwatch sw;

	if (id == 0) {
		readargs(argv, sz);
	}
	bcast(sz);
	
	arma::mat a = arma::randn(sz, sz);
	a += a.t();

	if (id == 0) {
		sw.run();
	}

	arma::vec eigval;
	arma::mat eigvec;
	arma::eig_sym(eigval, eigvec, a);

	if (id == 0) {
		sw.report();
	}

	MPI_Finalize();

	return 0;
}
