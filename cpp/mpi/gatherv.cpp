#include <mpi.h>
#include "../utility/mpi_helper.h"
#include "../utility/widgets.h"

int main(int, char** argv) {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	arma::uword sz_row, sz_col1_global = 0, sz_col2_global = 0;
	if (id == 0) {
		readargs(argv, sz_row, sz_col1_global, sz_col2_global);
	}

	bcast(sz_row, sz_col1_global, sz_col2_global);

	int sz_col1_local = sz_col1_global / nprocs;
	int sz_col2_local = sz_col2_global / nprocs;
	int rem1 = sz_col1_global % nprocs;
	int rem2 = sz_col2_global % nprocs;
	if (id < rem1)
		sz_col1_local += 1;
	if (id < rem2)
		sz_col2_local += 1;

	arma::mat global1, global2;
	arma::mat local1 = arma::ones(sz_row, sz_col1_local) * id;
	arma::mat local2 = arma::ones(sz_row, sz_col2_local) * id;

	if (id == 0) {
		global1.set_size(sz_row, sz_col1_global);
		global2.set_size(sz_row, sz_col2_global);
	}

	int status = gatherv(local1, global1, local2, global2);
	//int status = gatherv(local1, global1);

	if (id == 0) {
		global1.print();
		global2.print();
		std::cout << "status = " << status << std::endl;
	}

	::MPI_Finalize();


	return 0;
}
