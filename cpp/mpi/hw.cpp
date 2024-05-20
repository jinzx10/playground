#include <iostream>
#include <mpi.h>
#include "scalapack.h"

int main() {

	int id = 0, nprocs = 0;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);
	::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int icontext;

	Cblacs_get(0, 0, &icontext);
    std::cout << icontext << std::endl;

    Cblacs_gridinit(&icontext, "Row-major", 2, 2);

    int ictxt2;
	Cblacs_get(icontext, 10, &ictxt2);

    std::cout << ictxt2 << std::endl;



	::MPI_Finalize();

	return 0;
}
