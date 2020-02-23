// see if MPI_Comm_rank returns the same id for the same process during multiple calls
#include <mpi.h>
#include <iostream>

int main() {
	
	int id_first, id;
	MPI_Init(nullptr, nullptr);

	MPI_Comm_rank(MPI_COMM_WORLD, &id_first);

	for (int i = 0; i != 100; ++i) {
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		if (id_first == 1)
			std::cout << id << std::endl;
	}
	
	MPI_Finalize();

	return 0;
}
