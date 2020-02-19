#include <mpi.h>
#include <iostream>
#include "../utility/arma_mpi_helper.h"
#include "../utility/widgets.h"

int main(int, char** argv) {
	
	int id, nprocs;
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int x, y, z, w;
	arma::vec vx, vy, vz, vw;
	if (id == 0) {
		readargs(argv, x, y, z, w);
	}

	bcast(x, y, z, w);
	vx.set_size(x);
	vy.set_size(y);
	vz.set_size(z);
	vw.set_size(w);

	if (id == 0) {
		vx.zeros();
		vy.ones();
		vz.fill(2.0);
		vw.fill(3.0);
	}

	//bcast(vx, vy, vz, vw);

	if (id == 1) {
		std::cout << x << std::endl;
		std::cout << y << std::endl;
		std::cout << z << std::endl;
		std::cout << w << std::endl;
		vx.print();
		vy.print();
		vz.print();
		vw.print();
	}

	MPI_Finalize();


	return 0;
}

