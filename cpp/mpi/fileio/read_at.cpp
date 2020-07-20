#include <mpi.h>
#include <armadillo>

using namespace arma;

int main() {

	int id, nprocs;

	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	vec v(3,1);

	MPI_File fh;

	MPI_File_open(MPI_COMM_WORLD, "test.dat", MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
	MPI_Offset offset = id * 3 * sizeof(double); // not sizeof(MPI_DOUBLE)
	MPI_File_read_at(fh, offset, v.memptr(), 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	if (id == nprocs-1)
		v.print();
	

	MPI_Finalize();

	return 0;
}
