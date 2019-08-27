/* This program does the same thing as does dgesd2d.cpp
 * with all mkl subroutines.
 * link line and compiler options:
 *
 * -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_lp64 -lpthread -lm -ldl
 *
 * See intel link line advisor for help. */

#include <iostream>
#include <mpi.h>
#include <sstream>
#include <iomanip>
#include "../fstream/matio.h"
#include <mkl_blacs.h>
#include <mkl_scalapack.h>

int main(int argc, char** argv)
{
	::MPI_Init(nullptr, nullptr);

	char scope[] = "All";
	char grid_format[] = "Row";
	int ZERO = 0, ONE = 1;

	/* get process id and total process number */
	int ctxt, id, nprocs;
	::blacs_pinfo(&id, &nprocs);
	::blacs_get(0, &ZERO, &ctxt);

	/* initialize BLACS process grid (necessary for dgesd2d) */
	int ip_row, ip_col;
	::blacs_gridinit(&ctxt, grid_format, &ONE, &nprocs);
	::blacs_gridinfo(&ctxt, &ONE, &nprocs, &ip_row, &ip_col);
	std::cout << "id = " << id << "/" << nprocs << ", (" << ip_row << "," << ip_col << ")" << std::endl;

	::blacs_barrier(&ctxt, scope);

	/* source matrix */
	int sz_row_src = 11, sz_col_src = 9; // size of the source matrix
	int sz_row_dst = 7, sz_col_dst = 5; // size of the destination matrix
	double* A = nullptr; // source matrix
	if (!id) {
		A = new double[sz_row_src*sz_col_src];
		for (int i = 0; i < sz_row_src*sz_col_src; ++i) A[i] = i;

		std::cout << std::endl;
		std::cout << "source matrix" << std::endl;
		print_mat(A, sz_row_src, sz_col_src);
		std::cout << std::endl;
	}

	::blacs_barrier(&ctxt, scope);

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

	::blacs_barrier(&ctxt, scope);

	double* B = nullptr; // destination matrix
	if (id == p) {
		B = new double[sz_row_dst*sz_col_dst];
		for (int i = 0; i != sz_row_dst*sz_col_dst; ++i) B[i] = 0;
	}

	::blacs_barrier(&ctxt, scope);

	// send
	if (!id) {
		::dgesd2d(&ctxt, &M, &N, A+R+C*sz_row_src, &sz_row_src, &ZERO, &p);
	}

	// receive
	if (id == p) {
		::dgerv2d(&ctxt, &M, &N, B+r+c*sz_row_dst, &sz_row_dst, &ZERO, &ZERO);
	}

	if (id == p) {
		std::cout << std::endl;
		std::cout << "destination matrix (id = " << id << ")" << std::endl;
		print_mat(B, sz_row_dst, sz_col_dst);
		std::cout << std::endl;
	}

	::blacs_gridexit(&ctxt);
	::MPI_Finalize();

	return 0;
}
