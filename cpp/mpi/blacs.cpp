#include <armadillo>
#include <iostream>
#include <PBblacs.h>
#include <mpi.h>
#include <iostream>

int main() {
	///////////////////////////////////////////////////////
	//					MPI
	///////////////////////////////////////////////////////
	int mpi_rank, mpi_nprocs;
	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	::MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);

	int ctxt, blacs_rank, blacs_nprocs;
	char scope[1] = {'A'};

	::Cblacs_pinfo(&blacs_rank, &blacs_nprocs);
	::Cblacs_get(0, 0, &ctxt);
	std::cout << blacs_rank << std::endl;
	//::MPI_Barrier(MPI_COMM_WORLD);
	::Cblacs_barrier(ctxt, scope);
	std::cout << blacs_nprocs << std::endl;

	arma::mat A;
	arma::vec x;
	size_t sz;
	
	if (!blacs_rank) {
		sz = 1000;
		A = arma::ones(sz, sz);
		x = arma::ones(sz);
	}

	int nprow = 2, npcol = 2;
	char order[1] = {'R'};
	int irow, icol;

	::Cblacs_gridinit(&ctxt, order, nprow, npcol);
	::Cblacs_gridinfo(ctxt, &nprow, &npcol, &irow, &icol);





	::MPI_Finalize();

	return 0;
}
