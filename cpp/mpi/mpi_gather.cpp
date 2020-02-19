#include <mpi.h>
#include <iostream>

int main() {

	int id, nprocs;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int i = id;

	int* global = nullptr;

	if (id == 0) {
		global = new int[nprocs];
	}

	int status = MPI_Gather(&i, 1, MPI_INT, global, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (id == 0) {
		std::cout << "status = " << status << std::endl;
		for (int j = 0; j != nprocs; ++j)
			std::cout << global[j] << ' ';
		std::cout << std::endl;
	}


	int* larr = new int[id+1];
	for (int j = 0; j != id+1; ++j)
		larr[j] = id;

	int* nums = nullptr;
	int* garr = nullptr;
	int* disp = nullptr;

	if (id == 0) {
		garr = new int[nprocs*(nprocs+1)/2];
		nums = new int[nprocs];
		disp = new int[nprocs];

		for (int j = 0; j != nprocs; ++j) {
			nums[j] = j+1;
		}
		disp[0] = 0;
		for (int j = 1; j != nprocs; ++j) {
			disp[j] = disp[j-1]+nums[j-1];
		}
	}

	status = MPI_Gatherv(larr, id+1, MPI_INT, garr, nums, disp, MPI_INT, 0, MPI_COMM_WORLD);

	if (id == 0 ) {
		for (int j = 0; j != nprocs*(nprocs+1)/2; ++j) {
			std::cout << garr[j] << ' ';
		}
		std::cout << std::endl;
		std::cout << "status = " << status << std::endl;
	}

	MPI_Finalize();

	return 0;
}
