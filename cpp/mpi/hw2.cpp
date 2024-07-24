#include <iostream>
#include <mpi.h>

int main() {

	int id = 0, nprocs = 0;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	std::cout << "id = " << id << "/" << nprocs << std::endl;

	::MPI_Finalize();

	return 0;
}
