/* This test program demonstrates the basic usage of BLACS,
 * including grid initialization and data communication.
 * It will read a matrix from file, scatter its elements
 * to local processes, multiply by 2, and gather. */ 

#include <iostream>
#include <mpi.h>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include "scalapack.h"
#include "mpiaux.h"

int main(int argc, char** argv)
{
	::MPI_Init(nullptr, nullptr);

	/* get process id and total process number */
	int ctxt, id_blacs, np_blacs;
	Cblacs_pinfo(&id_blacs, &np_blacs);
	Cblacs_get(0, 0, &ctxt);

	/* command line input option check */
	if (argc < 8) {
		if (!id_blacs) {
			std::cout << "Usage: mpirun -np X ./blacs data.txt M N P Q m n" << std::endl
				<< "The program will read from data.txt a MxN matrix, use a PxQ process grid, and block size mxn " << std::endl;
		}
		::MPI_Barrier(MPI_COMM_WORLD);
		::MPI_Finalize();
		return 0;
	}

	/* initialize process grid */
	int np_row, np_col; // size of the process grid
	int ip_row, ip_col; // row and col index of the current process in the process grid
	if (!id_blacs) {
		// read the grid size from the command line
		std::stringstream ss;
		ss << argv[4] << ' ' << argv[5];
		ss >> np_row >> np_col;
	}
	// broadcast grid size
	::MPI_Bcast(&np_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
	::MPI_Bcast(&np_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// initialize the process grid
	char grid_format[] = "Row";
	Cblacs_gridinit(&ctxt, grid_format, np_row, np_col);
	Cblacs_gridinfo(ctxt, &np_row, &np_col, &ip_row, &ip_col);

	char scope[] = "All";
	Cblacs_barrier(ctxt, scope);

	/* read matrix from file */
	int sz_row, sz_col;
	double* A_glb = nullptr;

	if (!id_blacs) {
		std::stringstream ss;
		ss << argv[2] << ' ' << argv[3];
		ss >> sz_row >> sz_col;

		A_glb = new double[sz_row*sz_col];
		
		// read from file
		std::string filename = argv[1];
		std::ifstream file(filename.c_str());
		for (int irow = 0; irow != sz_row; ++irow) {
			for (int icol = 0; icol != sz_col; ++icol) {
				file >> A_glb[irow + icol*sz_row];
			}
		}

		std::cout << std::endl;
		std::cout << "raw matrix: " << std::endl;
		print_mat(A_glb, sz_row, sz_col);
		std::cout << std::endl;
	}

	// broadcast the size of the global matrix
	::MPI_Bcast(&sz_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
	::MPI_Bcast(&sz_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* scatter the global matrix */
	int sz_blk_row, sz_blk_col; // size of each block
	if (!id_blacs) {
		std::stringstream ss;
		ss << argv[6] << ' ' << argv[7];
		ss >> sz_blk_row >> sz_blk_col;
	}
	::MPI_Bcast(&sz_blk_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
	::MPI_Bcast(&sz_blk_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// local matrix
	double* A_loc = nullptr;

	// size of the local matrix
	int ZERO = 0; // auxiliary variable
	int sz_loc_row = numroc_(&sz_row, &sz_blk_row, &ip_row, &ZERO, &np_row);
	int sz_loc_col = numroc_(&sz_col, &sz_blk_col, &ip_col, &ZERO, &np_col);

	A_loc = new double[sz_loc_row*sz_loc_col];
	for (int i = 0; i != sz_loc_row*sz_loc_col; ++i)
		A_loc[i] = 0;

	std::cout << "id = " << id_blacs << "/" << np_blacs << ", ("
		<< ip_row << "," << ip_col << ")" << ", ("
		<< sz_loc_row << "x" << sz_loc_col << ")" << std::endl;

	// ready to communicate
	Cblacs_barrier(ctxt, scope);

	int sz_comm_row, sz_comm_col; // size of the communication block
	int proc_row = 0, proc_col = 0; // index of the process (in the process grid) that is responsible for the communication block
	int loc_r = 0, loc_c = 0; // local matrix indices
	for (int r = 0; r < sz_row; r += sz_blk_row, proc_row = (proc_row+1) % np_row) {
		sz_comm_row = (r + sz_blk_row <= sz_row) ? sz_blk_row : sz_row - r;
		proc_col = 0;
		for (int c = 0; c < sz_col; c += sz_blk_col, proc_col = (proc_col+1) % np_col) {
			sz_comm_col = (c + sz_blk_col <= sz_col) ? sz_blk_col : sz_col - c;
			if (!id_blacs) 
				Cdgesd2d(ctxt, sz_comm_row, sz_comm_col, A_glb+r+c*sz_row, sz_row, proc_row, proc_col);
			if (ip_row == proc_row && ip_col == proc_col) {
				Cdgerv2d(ctxt, sz_comm_row, sz_comm_col, A_loc+loc_r+loc_c*sz_loc_row, sz_loc_row, 0, 0);
				loc_c = (loc_c + sz_comm_col) % sz_loc_col;
			}
		}
		if (ip_row == proc_row)
			loc_r = (loc_r + sz_comm_row) % sz_loc_row;
	}

	Cblacs_barrier(ctxt, scope);

	/* print local matrices */
	for (int ip = 0; ip != np_blacs; ++ip) {
		Cblacs_barrier(ctxt, scope);
		if (id_blacs == ip) {
			std::cout << std::endl;
			std::cout << "local matrix at id = " << id_blacs << ": " << std::endl;
			print_mat(A_loc, sz_loc_row, sz_loc_col);
		}
		Cblacs_barrier(ctxt, scope);
	}

	Cblacs_barrier(ctxt, scope);

	// multiply local matrix by 2
	for (int i = 0; i != sz_loc_row*sz_loc_col; ++i)
		A_loc[i] *= 2;

	/* gather local matrices */
	double* B_glb = nullptr;
	if (!id_blacs) {
		B_glb = new double[sz_row*sz_col];
		for (int i = 0; i != sz_col*sz_row; ++i) B_glb[i] = 0;
	}

	Cblacs_barrier(ctxt, scope);

	loc_c = 0;
	loc_r = 0;
	proc_col = 0;
	proc_row = 0;
	for (int r = 0; r < sz_row; r += sz_blk_row, proc_row = (proc_row+1) % np_row) {
		sz_comm_row = (r + sz_blk_row <= sz_row) ? sz_blk_row : sz_row - r;
		proc_col = 0;
		for (int c = 0; c < sz_col; c += sz_blk_col, proc_col = (proc_col+1) % np_col) {
			sz_comm_col = (c + sz_blk_col <= sz_col) ? sz_blk_col : sz_col - c;
			if (ip_row == proc_row && ip_col == proc_col) {
				Cdgesd2d(ctxt, sz_comm_row, sz_comm_col, A_loc+loc_r+loc_c*sz_loc_row, sz_loc_row, 0, 0);
				loc_c = (loc_c + sz_comm_col) % sz_loc_col;
			}
			if (!id_blacs) 
				Cdgerv2d(ctxt, sz_comm_row, sz_comm_col, B_glb+r+c*sz_row, sz_row, proc_row, proc_col);
		}
		if (ip_row == proc_row)
			loc_r = (loc_r + sz_comm_row) % sz_loc_row;
	}

	Cblacs_barrier(ctxt, scope);

	if (!id_blacs) {
		std::cout << std::endl;
		std::cout << "matrix multiplied by 2: " << std::endl;
		print_mat(B_glb, sz_row, sz_col);
	}

	Cblacs_barrier(ctxt, scope);


	Cblacs_gridexit(ctxt);
	::MPI_Finalize();

	return 0;
}
