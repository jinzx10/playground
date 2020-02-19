#include <mpi.h>
#include "../utility/arma_mpi_helper.h"
#include "../utility/widgets.h"

int main(int, char** argv) {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int nglobal = 0;
	if (id == 0) {
		readargs(argv, nglobal);
	}

	bcast(&nglobal);

	int nlocal = nglobal / nprocs;
	int rem = nglobal % nprocs;
	if (id < rem) {
		nlocal += 1;
	}

	std::cout << "id = " << id << "   nlocal = " << nlocal << std::endl;

	arma::mat global;
	arma::mat local = arma::ones(3, nlocal) * id;

	if (id == 0) {
		global.set_size(3, nglobal);
	}

	gather(local, global);

	if (id == 0) {
		global.print();
	}


	::MPI_Finalize();


	return 0;
}
