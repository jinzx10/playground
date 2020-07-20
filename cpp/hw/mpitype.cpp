#include <mpi.h>
#include <iostream>

int main() {

	std::cout << "MPI_DOUBLE: " << typeid(MPI_DOUBLE).name() << std::endl;

	std::cout << "MPI_INT: " << MPI_INT << std::endl;
	std::cout << "MPI_UNSIGNED: " << MPI_UNSIGNED << std::endl;
	std::cout << "MPI_UNSIGNED_LONG_LONG: " << MPI_UNSIGNED_LONG_LONG << std::endl;
	std::cout << "MPI_DOUBLE: " << MPI_DOUBLE << std::endl;
	std::cout << "MPI_CHAR: " << MPI_CHAR << std::endl;


	return 0;
}
