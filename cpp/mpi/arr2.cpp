#include <mpi.h>
#include <iostream>

int main() {

	int num_procs, id;
	const size_t sz = 3;

	::MPI_Init(nullptr, nullptr);

	::MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);

	double* a = new double[sz];
	for (size_t i = 0; i < sz; ++i)
		a[i] = id + 0.1*id;

	// store all a
	double (*A)[sz] = nullptr;
	if (id == 0)
		A = new double[num_procs][sz];

	::MPI_Gather(a, sz, MPI_DOUBLE, A, sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (id == 0) {
		for (int i = 0; i < num_procs; ++i) {
			for (size_t j = 0; j < sz; ++j) {
				std::cout << A[i][j] << " "; 
			}
			std::cout << std::endl;
		}
	}

	delete [] a;
	delete [] A;

	::MPI_Finalize();
	return 0;
}
