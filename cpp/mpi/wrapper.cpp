#include <mpi.h>
#include <mkl_blacs.h>
#include <mkl_scalapack.h>
#include <armadillo>
#include <string>
#include "../utility/mpi_helper.h"
#include "helper.h"


int main() {

	int id, nprocs;
	int id_blacs, np_blacs, ctxt;
	int iZERO = 0;
	int iONE = 1;
	std::string sleep = "sleep 0.2";

	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	blacs_pinfo(&id_blacs, &np_blacs);
	blacs_get(&iZERO, &iZERO, &ctxt);
	
	/* initialize process grid */
	int np_row = 2, np_col = 2; // size of the process grid
	int ip_row, ip_col; // row and col index of the current process in the process grid
	
	// initialize the process grid
	char grid_format[] = "Row";
	blacs_gridinit(&ctxt, grid_format, &np_row, &np_col);
	blacs_gridinfo(&ctxt, &np_row, &np_col, &ip_row, &ip_col);

	char scope[] = "All";
	blacs_barrier(&ctxt, scope);

	arma::mat A_glb;
	int sz_row, sz_col, sz_row_blk, sz_col_blk;

	if (ip_row == 0 && ip_col == 0) {
		std::cout << "input global row size:" << std::endl;
		std::cin >> sz_row;
		std::cout << "input global col size:" << std::endl;
		std::cin >> sz_col;
		std::cout << "input block row size:" << std::endl;
		std::cin >> sz_row_blk;
		std::cout << "input block col size:" << std::endl;
		std::cin >> sz_col_blk;
		A_glb = arma::randu(sz_row, sz_col);
	}

	blacs_barrier(&ctxt, scope);
	bcast(sz_row, sz_col, sz_row_blk, sz_col_blk);

	int sz_row_loc = numroc(&sz_row, &sz_row_blk, &ip_row, &iZERO, &np_row);
	int sz_col_loc = numroc(&sz_col, &sz_col_blk, &ip_col, &iZERO, &np_col);

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
			std::cout << "mpi id = " << id << "/" << nprocs << std::endl
				<< "blacs id = (" << ip_row << "," << ip_col <<")" << std::endl
				<< "block size = (" << sz_row_blk << " x " << sz_col_blk << ")" << std::endl
				<< "local matrix size = (" << sz_row_loc << " x " << sz_col_loc << ")" << std::endl
				<< std::endl;
		}
		std::system(sleep.c_str());
	}

	arma::mat A_loc = arma::zeros(sz_row_loc, sz_col_loc);

	if (ip_row == 0 && ip_col == 0) {
		dgesd2d(&ctxt, &sz_row_blk, &sz_col_blk, A_glb.memptr()+1, &sz_row, &iZERO, &iONE);
	}
	
	if (ip_row == 0 && ip_col == 1) {
		dgerv2d(&ctxt, &sz_row_blk, &sz_col_blk, A_loc.memptr(), &sz_row_loc, &iZERO, &iZERO);
	}

	blacs_barrier(&ctxt, scope);

	if (ip_row == 0 && ip_col == 0) {
		std::cout << "global matrix = " << std::endl;
		A_glb.print();
	}

	std::system(sleep.c_str());

	if (ip_row == 0 && ip_col == 1) {
		std::cout << "(" << ip_row << "x" << ip_col 
			<< ") local matrix = " << std::endl;
		A_loc.print();
	}


	blacs_gridexit(&ctxt);
	MPI_Finalize();

}
