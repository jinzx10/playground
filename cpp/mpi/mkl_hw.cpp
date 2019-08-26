#include <iostream>
#include <mpi.h>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <mkl_blacs.h>
#include <mkl_scalapack.h>

void print(double const* A, int sz_row, int sz_col, int width = 4) {
	for (int r = 0; r < sz_row; ++r) {
		for (int c = 0; c < sz_col; ++c) {
			std::cout << std::setw(width) << A[r*sz_col+c] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char** argv)
{
	::MPI_Init(nullptr, nullptr);

	//int mpi_id, mpi_nprocs;
	//::MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
	//::MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs);

	/* get process id and total process number by Cblacs_pinfo */
	int ctxt, id_blacs, np_blacs;
	char scope[] = "All";
	blacs_pinfo(&id_blacs, &np_blacs);
	blacs_get(0, 0, &ctxt);
	//std::cout << "id = " << id_blacs << "/" << np_blacs << std::endl;

	/* command line input check */
//	if (argc < 6) {
//		if (!id_blacs) {
//			std::cerr << "Usage: mpirun -np X ./blacs data.txt M N P Q" << std::endl
//				<< "will read from data.txt a MxN matrix and use a PxQ process grid" << std::endl;
//		}
//		::MPI_Barrier(MPI_COMM_WORLD);
//		::MPI_Finalize();
//		return -1;
//	}
//
//	/* initialize process grid */
//	int np_row, np_col; // size of the process grid
//	int ip_row, ip_col; // row and col index of the current process in the process grid
//	if (!id_blacs) {
//		// read the grid size from the command line
//		std::stringstream ss;
//		ss << argv[4] << ' ' << argv[5];
//		ss >> np_row >> np_col;
//	}
//	// broadcast grid size
//	::MPI_Bcast(&np_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
//	::MPI_Bcast(&np_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//	// initialize the process grid
//	char grid_format[] = "Row";
//	blacs_gridinit(&ctxt, grid_format, &np_row, &np_col);
//	blacs_gridinfo(&ctxt, &np_row, &np_col, &ip_row, &ip_col);
//
//	blacs_barrier(&ctxt, scope);
//	//::MPI_Barrier(MPI_COMM_WORLD);
//
//	/* read matrix from file */
//	int sz_row, sz_col;
//	double* A_glb = nullptr;
//
//	if (!id_blacs) {
//		std::stringstream ss;
//		ss << argv[2] << ' ' << argv[3];
//		ss >> sz_row >> sz_col;
//
//		A_glb = new double[sz_row*sz_col];
//		
//		// read from file
//		std::string filename = argv[1];
//		std::ifstream file(filename.c_str());
//		for (int irow = 0; irow != sz_row; ++irow) {
//			for (int icol = 0; icol != sz_col; ++icol) {
//				file >> A_glb[irow*sz_col + icol];
//			}
//		}
//
//		print(A_glb, sz_row, sz_col);
//		std::cout << std::endl;
//	}
//
//	// broadcast the size of the global matrix
//	::MPI_Bcast(&sz_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
//	::MPI_Bcast(&sz_col, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//	/* scatter the global matrix */
//
//	// size of the local matrix
//	int ZERO = 0; // auxiliary variable
//
//	// ready to communicate
//	blacs_barrier(&ctxt, scope);
//
//	double* B = new double[sz_row*sz_col];
//	for (int i = 0; i < sz_row*sz_col; ++i) B[i] = 0;
//
//	int nrow_comm = 3;
//	int ncol_comm = 2;
//	if (!id_blacs) {
//		dgesd2d(&ctxt, &nrow_comm, &ncol_comm, A_glb + 12, &sz_col, &ZERO, &ZERO);
//		dgerv2d(&ctxt, &nrow_comm, &ncol_comm, B+3, &sz_col, &ZERO, &ZERO);
//	}
//
//	/*
//	int sz_comm_row, sz_comm_col; // size of the communication block
//	int proc_row = 0, proc_col = 0; // index of the process (in the process grid) that is responsible for the communication block
//	int loc_r = 0, loc_c = 0; // local matrix indices
//	for (int r = 0; r < sz_row; r += sz_blk_row, proc_row = (proc_row+1) % np_row) {
//		sz_comm_row = (r + sz_blk_row <= sz_row) ? sz_blk_row : sz_row - r;
//		for (int c = 0; c < sz_col; c += sz_blk_col, proc_col = (proc_col+1) % np_col) {
//			sz_comm_col = (c + sz_blk_col <= sz_col) ? sz_blk_col : sz_col - c;
//			if (!id_blacs) 
//				Cdgesd2d(ctxt, sz_comm_row, sz_comm_col, A_glb+r*sz_col+c, sz_row, proc_row, proc_col);
//
//			if (ip_row == proc_row && ip_col == proc_col) {
//				Cdgerv2d(ctxt, sz_comm_row, sz_comm_col, A_loc+loc_r*sz_loc_col+loc_c, sz_loc_row, 0, 0);
//				loc_c = (loc_c + sz_comm_col) % sz_loc_col;
//			}
//		}
//		if (ip_row == proc_row)
//			loc_r = (loc_r + sz_comm_row) % sz_loc_row;
//	}
//	*/
//
//
//	blacs_barrier(&ctxt, scope);
//
//	if (!id_blacs) {
//		std::cout << std::endl;
//		print(B, sz_row, sz_col);
//	}
//
//	blacs_barrier(&ctxt, scope);
//
//
//	blacs_gridexit(&ctxt);
	::MPI_Finalize();

	return 0;
}
