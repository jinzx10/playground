#include <mpi.h>
#include <iostream>

struct Test
{
	int i[2];
	char c[3];
	double d[5];
};

int main() {

	MPI_Init(nullptr, nullptr);

	int id, nprocs;

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	MPI_Datatype MPI_Test_t;
	MPI_Datatype types[3] = {MPI_INT, MPI_CHAR, MPI_DOUBLE};
	int blocklens[] = {2, 3, 5};
	MPI_Aint disps[3] = {offsetof(Test, i), offsetof(Test, c), offsetof(Test, d)};

	::MPI_Type_create_struct(3, blocklens, disps, types, &MPI_Test_t);
	::MPI_Type_commit(&MPI_Test_t);

	Test t[3];

	if (id == 0) {
		for (int it = 0; it != 3; ++it) {
			for (int ii = 0; ii != 2; ++ii) {
				t[it].i[ii] = it*10 + ii;
			}
			for (int ic = 0; ic != 3; ++ic) {
				t[it].c[ic] = 65+ic;
			}
			for (int id = 0; id != 5; ++id) {
				t[it].d[id] = it*10 + 0.1*id;
			}
		}
	}

	::MPI_Bcast(t, 3, MPI_Test_t, 0, MPI_COMM_WORLD);

	if (id == 1) {
		for (auto& e : t[2].d)
			std::cout << e << std::endl;
	}

	MPI_Finalize();
	return 0;
}
