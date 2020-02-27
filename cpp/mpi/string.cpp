#include <mpi.h>
#include "../utility/mpi_helper.h"

int main() {

	int id, nprocs;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	std::string dir;
	int sz;
	double val;
	if (id == 0) {
		dir = "/home/zuxin/playground/cpp";
		sz = 5;
		val = 3.14;
	}

	bcast(dir, sz, val);

	for (int i = 1; i != nprocs; ++i) {
		if (id == i) {
			std::cout << dir << std::endl;
			std::cout << sz << std::endl;
			std::cout << val << std::endl;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		std::system("sleep 0.1");
	}

	MPI_Finalize();

	return 0;
}
