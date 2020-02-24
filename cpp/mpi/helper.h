#ifndef __HELPER_H__
#define __HELPER_H__

#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <armadillo>
#include "../utility/mpi_helper.h"


// scatter A of process (ip_row_root, ip_col_root) to A_loc of each process
// A_loc should be pre-allocated
// the function assumes the column-major layout 
inline void scatter(
		int		const&		ctxt, 
		double*				A,
		double*				A_loc,
		int		const&		sz_row,
		int 	const& 		sz_col,
		int 	const& 		sz_row_blk,
		int 	const& 		sz_col_blk,
		int 	const& 		ip_row,
		int 	const& 		ip_col,
		int 	const& 		np_row,
		int 	const& 		np_col,
		int 	const& 		ip_row_root = 0,
		int 	const& 		ip_col_root = 0,
		int 	const& 		ip_row_start = 0,
		int 	const& 		ip_col_start = 0
) {
	int sz_row_loc = numroc(&sz_row, &sz_row_blk, &ip_row, &ip_row_start, &np_row);
	int sz_col_loc = numroc(&sz_col, &sz_col_blk, &ip_col, &ip_col_start, &np_col);

	// grid indices of the block to communicate
	int ipr = ip_row_start, ipc = ip_col_start;

	// size of the block to communicate
	int szr = 0, szc = 0; 

	// indices in A_loc
	int r_loc = 0, c_loc = 0; 

	for (int r = 0; r < sz_row; r += sz_row_blk, ipr = (ipr+1)%np_row) {
		szr = (r + sz_row_blk > sz_row) ? sz_row - r : sz_row_blk;
		for (int c = 0; c < sz_col; c += sz_col_blk, ipc = (ipc+1)%np_col) {
			szc = (c + sz_col_blk > sz_col) ? sz_col - c : sz_col_blk;
			if (ip_row == ip_row_root && ip_col == ip_col_root) 
				dgesd2d(&ctxt, &szr, &szc, A+r+c*sz_row, &sz_row, &ipr, &ipc);
			if (ip_row == ipr && ip_col == ipc) {
				dgerv2d(&ctxt, &szr, &szc, A_loc+r_loc+c_loc*sz_row_loc, &sz_row_loc, &ip_row_root, &ip_col_root);
				c_loc = (c_loc + szc) % sz_col_loc;
			}
		}
		ipc = ip_col_start;
		if (ip_row == ipr)
			r_loc += szr;
	}
}


// gather A_loc of each process to A of process (ip_row_root, ip_col_root)
// A should be pre-allocated
// the function assumes the column-major layout 
inline void gather(
		int		const&		ctxt, 
		double*				A,
		double*				A_loc,
		int		const&		sz_row,
		int 	const& 		sz_col,
		int 	const& 		sz_row_blk,
		int 	const& 		sz_col_blk,
		int 	const& 		ip_row,
		int 	const& 		ip_col,
		int 	const& 		np_row,
		int 	const& 		np_col,
		int 	const& 		ip_row_root = 0,
		int 	const& 		ip_col_root = 0,
		int 	const& 		ip_row_start = 0,
		int 	const& 		ip_col_start = 0
) {
	int sz_row_loc = numroc(&sz_row, &sz_row_blk, &ip_row, &ip_row_start, &np_row);
	int sz_col_loc = numroc(&sz_col, &sz_col_blk, &ip_col, &ip_col_start, &np_col);

	// grid indices of the block to communicate
	int ipr = ip_row_start, ipc = ip_col_start;

	// size of the block to communicate
	int szr = 0, szc = 0; 

	// indices in A_loc
	int r_loc = 0, c_loc = 0; 

	for (int r = 0; r < sz_row; r += sz_row_blk, ipr = (ipr+1)%np_row) {
		szr = (r + sz_row_blk > sz_row) ? sz_row - r : sz_row_blk;
		for (int c = 0; c < sz_col; c += sz_col_blk, ipc = (ipc+1)%np_col) {
			szc = (c + sz_col_blk > sz_col) ? sz_col - c : sz_col_blk;
			if (ip_row == ipr && ip_col == ipc) {
				dgesd2d(&ctxt, &szr, &szc, A_loc+r_loc+c_loc*sz_row_loc, &sz_row_loc, &ip_row_root, &ip_col_root);
				c_loc = (c_loc + szc) % sz_col_loc;
			}
			if (ip_row == ip_row_root && ip_col == ip_col_root)
				dgerv2d(&ctxt, &szr, &szc, A+r+c*sz_row, &sz_row, &ipr, &ipc);
		}
		ipc = ip_col_start;
		if (ip_row == ipr)
			r_loc += szr;
	}
}


void pmatmul(arma::mat& A, arma::mat& B, arma::mat& C) {
	int iZERO = 0, iONE = 1;
	double dZERO = 0.0, dONE = 1.0;

	int ctxt, id_blacs, np_blacs;
	blacs_pinfo(&id_blacs, &np_blacs);
	blacs_get(&iZERO, &iZERO, &ctxt);

	//int np_row = sqrt(np_blacs), np_col = sqrt(np_blacs);
	int np_row, np_col;
	for (np_row = sqrt(np_blacs); np_row != 1; --np_row) {
		if (np_blacs % np_row == 0)
			break;
	}
	np_col = np_blacs / np_row;

	int ip_row, ip_col;
	char layout = 'C';
	blacs_gridinit(&ctxt, &layout, &np_row, &np_col);
	blacs_gridinfo(&ctxt, &np_row, &np_col, &ip_row, &ip_col);

	int id, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int szA_row, szA_col, szA_row_blk, szA_col_blk, szA_row_loc, szA_col_loc;
	int szB_row, szB_col, szB_row_blk, szB_col_blk, szB_row_loc, szB_col_loc;
	int szC_row, szC_col, szC_row_blk, szC_col_blk, szC_row_loc, szC_col_loc;
	if (id == 0) {
		szA_row = A.n_rows;
		szA_col = A.n_cols;
		szA_row_blk = szA_row / np_row + (szA_row % np_row != 0);
		szA_col_blk = szA_col / np_col + (szA_col % np_col != 0);

		szB_row = B.n_rows;
		szB_col = B.n_cols;
		szB_row_blk = szB_row / np_row + (szB_row % np_row != 0);
		szB_col_blk = szB_col / np_col + (szB_col % np_col != 0);

		C.set_size(A.n_rows, B.n_cols);
		szC_row = C.n_rows;
		szC_col = C.n_cols;
		szC_row_blk = szC_row / np_row + (szC_row % np_row != 0);
		szC_col_blk = szC_col / np_col + (szC_col % np_col != 0);
	}

	bcast(  szA_row, szA_col, szA_row_blk, szA_col_blk,
			szB_row, szB_col, szB_row_blk, szB_col_blk,
			szC_row, szC_col, szC_row_blk, szC_col_blk  );

	szA_row_loc = numroc(&szA_row, &szA_row_blk, &ip_row, &iZERO, &np_row);
	szA_col_loc = numroc(&szA_col, &szA_col_blk, &ip_col, &iZERO, &np_col);
	szB_row_loc = numroc(&szB_row, &szB_row_blk, &ip_row, &iZERO, &np_row);
	szB_col_loc = numroc(&szB_col, &szB_col_blk, &ip_col, &iZERO, &np_col);
	szC_row_loc = numroc(&szC_row, &szC_row_blk, &ip_row, &iZERO, &np_row);
	szC_col_loc = numroc(&szC_col, &szC_col_blk, &ip_col, &iZERO, &np_col);

	int descA[9], descB[9], descC[9];
	int info;
	descinit(descA, &szA_row, &szA_col, &szA_row_blk, &szA_col_blk, &iZERO, &iZERO, &ctxt, &szA_row_loc, &info);
	descinit(descB, &szB_row, &szB_col, &szB_row_blk, &szB_col_blk, &iZERO, &iZERO, &ctxt, &szB_row_loc, &info);
	descinit(descC, &szC_row, &szC_col, &szC_row_blk, &szC_col_blk, &iZERO, &iZERO, &ctxt, &szC_row_loc, &info);

	arma::mat A_loc(szA_row_loc, szA_col_loc);
	arma::mat B_loc(szB_row_loc, szB_col_loc);
	arma::mat C_loc(szC_row_loc, szC_col_loc);

	scatter(ctxt, A.memptr(), A_loc.memptr(), szA_row, szA_col, szA_row_blk, szA_col_blk, ip_row, ip_col, np_row, np_col);
	scatter(ctxt, B.memptr(), B_loc.memptr(), szB_row, szB_col, szB_row_blk, szB_col_blk, ip_row, ip_col, np_row, np_col);

#ifdef PRINT
	if (id == 0) {
		std::cout << "A = " << std::endl;
		A.print();
		std::cout << std::endl;
	}
	std::system("sleep 0.2");
	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
			std::cout << "id = " << id << "   A_loc = " << std::endl;
			A_loc.print();
			std::cout << std::endl;
		}
		std::system("sleep 0.2");
	}
	std::system("sleep 0.2");
	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		std::cout << "B = " << std::endl;
		B.print();
		std::cout << std::endl;
	}
	std::system("sleep 0.2");
	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
			std::cout << "id = " << id << "   B_loc = " << std::endl;
			B_loc.print();
		}
		std::system("sleep 0.2");
	}
	std::system("sleep 0.2");
	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
			std::cout << "id = " << id << "   C_loc = " << std::endl;
			C_loc.print();
		}
		std::system("sleep 0.2");
	}
	std::cout << std::endl;

	std::system("sleep 0.2");
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	char trans = 'N';
	pdgemm(&trans, &trans, &szA_row, &szB_col, &szA_col, &dONE, A_loc.memptr(), &iONE, &iONE, descA, B_loc.memptr(), &iONE, &iONE, descB, &dZERO, C_loc.memptr(), &iONE, &iONE, descC);

	gather(ctxt, C.memptr(), C_loc.memptr(), szC_row, szC_col, szC_row_blk, szC_col_blk, ip_row, ip_col, np_row, np_col);

	blacs_gridexit(&ctxt);
}






#endif
