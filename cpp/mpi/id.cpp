// see in multiple calls if MPI_Comm_rank and blacs_pinfo returns the same id for the same process 
#include <mpi.h>
#include <mkl_blacs.h>
#include <iostream>

int main() {
	
	int id_first, id;
	int id_blacs_first, id_blacs, np_blacs;
	MPI_Init(nullptr, nullptr);

	MPI_Comm_rank(MPI_COMM_WORLD, &id_first);
	blacs_pinfo(&id_blacs_first, &np_blacs);

	for (int i = 0; i != 10; ++i) {
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		blacs_pinfo(&id_blacs, &np_blacs);
		for (int j = 0; j != np_blacs; ++j) {
			if (j == id_first)
				std::cout << id_first << " " << id << " " << id_blacs_first << " " << id_blacs << std::endl;
			MPI_Barrier(MPI_COMM_WORLD);
			std::system("sleep 0.1");
		}
	}
	
	MPI_Finalize();

	return 0;
}
