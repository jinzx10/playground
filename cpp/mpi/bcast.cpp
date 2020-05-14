#include <mpi.h>
#include <iostream>
#include "../utility/mpi_helper.h"
#include "../utility/widgets.h"

int main(int, char** argv) {
	
	int id, nprocs;
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int root;
	int sz;

	if (id == 0) {
		readargs(argv, root);
	}
	bcast(0, root); // broadcast which proc is root


	arma::vec v = id*arma::ones(id);
	sz = id;
	bcast(root, sz);

	v.set_size(sz);

	if (id == 0) {
		v.print();
	}


	MPI_Finalize();


	return 0;
}

