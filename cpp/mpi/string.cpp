#include <mpi.h>
#include "../utility/mpi_helper.h"

int main() {

	int id, nprocs;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	std::string dir;
	if (id == 0) {
		dir = "/home/zuxin/playground/cpp";
	}

	bcast(dir);

	if (id == 1) {
		std::cout << dir << std::endl;
	}

	MPI_Finalize();

	return 0;
}
