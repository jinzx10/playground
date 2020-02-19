#include <mpi.h>
#include "../utility/arma_mpi_helper.h"
#include "../utility/widgets.h"

int main(int, char** argv) {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	arma::uword nglobal = 0;
	if (id == 0) {
		readargs(argv, nglobal);
	}

	bcast(&nglobal);

	int nlocal = nglobal / nprocs;
	int rem = nglobal % nprocs;
	if (id < rem) {
		nlocal += 1;
	}
	arma::Col<int> n_each_local, disp;
   
	if (id == 0) {
		n_each_local = arma::zeros<arma::Col<int>>(nprocs);
	}

	gather(&nlocal, n_each_local);

	if (id == 0) {
		n_each_local*=3;
		disp = arma::cumsum(n_each_local);
		disp.tail(disp.n_elem-1) = disp.head(disp.n_elem-1);
		disp(0) = 0;
	}

	if (id == 0) {
		n_each_local.print();
		disp.print();
	}

	arma::mat global;
	arma::mat local = arma::ones(3, nlocal) * id;

	if (id == 0) {
		global.set_size(3, nglobal);
	}

	//::MPI_Gatherv(local.memptr(), local.n_elem, MPI_DOUBLE, global.memptr(), n_each_local.memptr(), disp.memptr(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//gatherv(local, n_each_local, global);
	gatherv(local, global);

	if (id == 0) {
		global.print();
	}


	::MPI_Finalize();


	return 0;
}
