#include <mpi.h>
#include <armadillo>
#include <iostream>

using namespace arma;

int main() {

	int num_procs, id;
	int sz = 3;

	::MPI_Init(nullptr, nullptr);

	::MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);

	//arma::arma_rng::set_seed(id);
	arma::arma_rng::set_seed_random();
	mat a = randu(sz,sz) + ones(sz,sz)*id;
	cube A; 
	mat B;

	if (id == 0) {
		A = zeros(sz, sz, num_procs);
		B = zeros(sz, sz);
	}

	::MPI_Gather(a.memptr(), sz*sz, MPI_DOUBLE, A.memptr(), sz*sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Reduce(a.memptr(), B.memptr(), sz*sz, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (id == 0) {
		A.print();
		arma::sum(A, 2).print();
		B.print();
	}

	::MPI_Finalize();
	return 0;
}
