#ifndef __MPI_AUX_H__
#define __MPI_AUX_H__

#include <mpi.h>
#include <iostream>
#include <iomanip>

void mpi_start(int* id, int* nprocs) {
	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_rank(MPI_COMM_WORLD, id);
	::MPI_Comm_size(MPI_COMM_WORLD, nprocs);
}


void print_id(int* id, int* nprocs) {
	std::cout << "id = " << *id << "/" << *nprocs << std::endl;
}


void print_mat(double const* const& A, int const& sz_row, int const& sz_col, bool column_major = true, int const& width = 4) {
	for (int r = 0; r != sz_row; ++r) {
		for (int c = 0; c != sz_col; ++c) {
			std::cout << std::setw(width) << (column_major ? A[r+c*sz_row] : A[r*sz_col+c] ) << " ";
		}
		std::cout << std::endl;
	}
}

void mpi_end() {
	::MPI_Finalize();
}

#endif
