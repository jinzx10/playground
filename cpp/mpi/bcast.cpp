#include <mpi.h>
#include <iostream>
#include "../utility/mpi_helper.h"
#include "../utility/widgets.h"

int main(int, char** argv) {
	
	int id, nprocs;
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int x, y;
	arma::vec vx, vy;
	if (id == 0) {
		readargs(argv, x, y);
	}

	int status = bcast(x, y);
	if (id == 0) {
		std::cout << "bcast(x,y) return value = " << status << std::endl;
	}

	vx.set_size(x);
	vy.set_size(y);

	if (id == 0) {
		vx.zeros();
		vy.ones();
		x = 321;
		y = 123;
	}

	bcast(vx, x, y, vy);

	if (id == nprocs-1) {
		std::cout << x << std::endl;
		std::cout << y << std::endl;
		vx.print();
		vy.print();
	}

	MPI_Finalize();


	return 0;
}

