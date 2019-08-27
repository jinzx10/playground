/* This program shows the standard usage of Cdgesd2d and Cdgerv2d.
 * Specifically, a block of the source matrix in root process will
 * be sent to a specific process at a specific position in the 
 * destination matrix.
 * These BLACS routines are probably C-wrappers of fortran programs
 * and thus naturally work with column-major storage. */

#include <iostream>
#include <mpi.h>
#include <sstream>
#include <iomanip>
#include "scalapack.h"

void print(double const* A, int sz_row, int sz_col, int width = 4) {
	for (int r = 0; r < sz_row; ++r) {
		for (int c = 0; c < sz_col; ++c) {
			// interpret 1d-array as column-major matrix
			std::cout << std::setw(width) << A[r+c*sz_row] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char** argv)
{
	::MPI_Init(nullptr, nullptr);

	/* get process id and total process number */
	int ctxt, id, nprocs;
	Cblacs_pinfo(&id, &nprocs);
	Cblacs_get(0, 0, &ctxt);

	/* initialize BLACS process grid (necessary for Cdgesd2d) */
	char grid_format[] = "Row";
	char scope[] = "All";
	int ONE = 1;
	int ip_row, ip_col;
	Cblacs_gridinit(&ctxt, grid_format, ONE, nprocs);
	Cblacs_gridinfo(ctxt, &ONE, &nprocs, &ip_row, &ip_col);
	std::cout << "id = " << id << "/" << nprocs << ", (" << ip_row << "," << ip_col << ")" << std::endl;

	Cblacs_barrier(ctxt, scope);

	/* source matrix */
	int sz_row_src = 11, sz_col_src = 9; // size of the source matrix
	int sz_row_dst = 7, sz_col_dst = 5; // size of the destination matrix
	double* A = nullptr; // source matrix
	if (!id) {
		A = new double[sz_row_src*sz_col_src];
		for (int i = 0; i < sz_row_src*sz_col_src; ++i) A[i] = i;

		std::cout << std::endl;
		std::cout << "source matrix" << std::endl;
		print(A, sz_row_src, sz_col_src);
		std::cout << std::endl;
	}

	Cblacs_barrier(ctxt, scope);

	/* command line input check */
	if (argc < 8) {
		if (!id) {
			std::cerr << "Usage: mpirun -np X ./Cdgesd2d M N R C p r c" << std::endl
				<< "will send a MxN block of source matrix at (R,C) to (r,c) of the destination matrix at process p" << std::endl;
		}
		::MPI_Barrier(MPI_COMM_WORLD);
		::MPI_Finalize();
		return -1;
	}

	int params[7];
	if (!id) {
		std::stringstream ss;
		ss << argv[1] << ' ' << argv[2] << ' ' << argv[3] << ' '<< argv[4] << ' '
			<< argv[5] << ' ' << argv[6] << ' ' << argv[7];
		ss >> params[0] >> params[1] >> params[2] >> params[3] >> params[4] >> params[5] >> params[6]; 
	}
	::MPI_Bcast(params, 7, MPI_INT, 0, MPI_COMM_WORLD);
	int M = params[0], N = params[1], R = params[2], C = params[3], p = params[4], r = params[5], c = params[6];

	// overflow check
	if (R+M > sz_row_src || C+N > sz_col_src) {
		if (!id)
			std::cerr << "source overflow" << std::endl;
		::MPI_Barrier(MPI_COMM_WORLD);
		::MPI_Finalize();
		return -1;
	}

	if (r+M > sz_row_dst || c+N > sz_col_dst) {
		if (!id)
			std::cerr << "destination overflow" << std::endl;
		::MPI_Barrier(MPI_COMM_WORLD);
		::MPI_Finalize();
		return -1;
	}

	if (p >= nprocs) {
		if (!id)
			std::cerr << "process id overflow" << std::endl;
		::MPI_Barrier(MPI_COMM_WORLD);
		::MPI_Finalize();
		return -1;
	}
	
	if (!id)
		std::cout << M << "x" << N << ", " << "(" << R << "," << C << ") " << "-->" << " (" << r << "," << c << ")" <<  std::endl;

	Cblacs_barrier(ctxt, scope);

	double* B = nullptr; // destination matrix
	if (id == p) {
		B = new double[sz_row_dst*sz_col_dst];
		for (int i = 0; i != sz_row_dst*sz_col_dst; ++i) B[i] = 0;
	}

	Cblacs_barrier(ctxt, scope);

	// send
	if (!id) {
		Cdgesd2d(ctxt, M, N, A+R+C*sz_row_src, sz_row_src, 0, p);
	}

	// receive
	if (id == p) {
		Cdgerv2d(ctxt, M, N, B+r+c*sz_row_dst, sz_row_dst, 0, 0);
	}

	if (id == p) {
		std::cout << std::endl;
		std::cout << "destination matrix (id = " << id << ")" << std::endl;
		print(B, sz_row_dst, sz_col_dst);
		std::cout << std::endl;
	}

	Cblacs_gridexit(ctxt);
	::MPI_Finalize();

	return 0;
}
