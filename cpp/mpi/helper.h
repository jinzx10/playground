#ifndef __HELPER_H__
#define __HELPER_H__

#include <mkl_blacs.h>
#include <mkl_scalapack.h>
#include <armadillo>

void pmatmul(arma::mat& A, arma::mat& B, arma::mat& C) {
	int iZERO = 0;
	int iONE = 1;
	int ctxt, id_blacs, np_blacs;
	blacs_pinfo(&id_blacs, &np_blacs);
	blacs_get(&iZERO, &iZERO, &ctxt);

	int np_row, np_col;
	int ip_row, ip_col;
	char layout = 'C';
	blacs_gridinit(&ctxt, &layout, &np_row, &np_col);
	blacs_gridinfo(&ctxt, &np_row, &np_col, &ip_row, &ip_col);

	int sz_row = A.n_rows;
	int sz_col = A.n_cols;
	int sz_row_blk = 2;
	int sz_col_blk = 2;
	int sz_row_loc = numroc(&sz_row, &sz_row_blk, &ip_row, &iZERO, &np_row);
	int sz_col_loc = numroc(&sz_col, &sz_col_blk, &ip_col, &iZERO, &np_col);

	arma::mat A_loc = arma::zeros(sz_row_loc, sz_col_loc);

	if (ip_row == 0 && ip_col == 0) {
		dgesd2d(&ctxt, &sz_row_blk, &sz_col_blk, A.memptr()+1, &sz_row, &iZERO, &iONE);
	}
	
	if (ip_row == 0 && ip_col == 1) {
		dgerv2d(&ctxt, &sz_row_blk, &sz_col_blk, A_loc.memptr(), &sz_row_loc, &iZERO, &iZERO);
	}

	if (id_blacs == 0) {
		B.zeros(sz_row, sz_col);
	}
}





#endif
