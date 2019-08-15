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
	vec a = randu(sz) + ones(sz)*id;
	mat A; 
	mat B;

	if (id == 0) {
		A = zeros(sz, num_procs);
		B = zeros(sz-1, num_procs);
	}

	::MPI_Gather(a.memptr(), sz, MPI_DOUBLE, A.memptr(), sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Gather(a.begin_row(1), sz-1, MPI_DOUBLE, B.memptr(), sz-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		A.print();
		std::cout << std::endl;
		B.print();
	}

	::MPI_Finalize();
	return 0;
}
