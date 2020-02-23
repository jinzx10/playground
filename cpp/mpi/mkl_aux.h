#ifndef __MKL_AUX_H__
#define __MKL_AUX_H__

#include <mkl_blacs.h>
#include <mkl_scalapack.h>

// scatter A in process (src_row, src_col) to A_loc in each process
inline void scatter(
		int		const&		ctxt, 
		double*		&		A,
		double*		&		A_loc,
		int		const&		sz_row,
		int 	const& 		sz_col,
		int 	const& 		sz_blk_row,
		int 	const& 		sz_blk_col,
		int 	const& 		ip_row,
		int 	const& 		ip_col,
		int 	const& 		np_row,
		int 	const& 		np_col,
		int 	const& 		src_row = 0,
		int 	const& 		src_col = 0,
		int 	const& 		ip_row_start = 0,
		int 	const& 		ip_col_start = 0
);

// gather A_loc in each process to A in process (src_row, src_col)
inline void gather(
		int		const&		ctxt, 
		double*		&		A,
		double*		&		A_loc,
		int		const&		sz_row,
		int 	const& 		sz_col,
		int 	const& 		sz_blk_row,
		int 	const& 		sz_blk_col,
		int 	const& 		ip_row,
		int 	const& 		ip_col,
		int 	const& 		np_row,
		int 	const& 		np_col,
		int 	const& 		src_row = 0,
		int 	const& 		src_col = 0,
		int 	const& 		ip_row_start = 0,
		int 	const& 		ip_col_start = 0
);

inline void scatter(int const& ctxt, double*& A, double*& A_loc, int const& sz_row, int const& sz_col, int const& sz_blk_row, int const& sz_blk_col, int const& ip_row, int const& ip_col, int const& np_row, int const& np_col, int const& src_row, int const& src_col, int const& ip_row_start, int const& ip_col_start) {
	int sz_loc_row = ::numroc(&sz_row, &sz_blk_row, &ip_row, &ip_row_start, &np_row);
	int sz_loc_col = ::numroc(&sz_col, &sz_blk_col, &ip_col, &ip_col_start, &np_col);

	delete[] A_loc;
	A_loc = new double[sz_loc_row*sz_loc_col];

	int pid_row = ip_row_start, pid_col = ip_col_start;
	int sz_comm_row = 0, sz_comm_col = 0;
	int r_loc = 0, c_loc = 0;
	for (int r = 0; r < sz_row; r += sz_blk_row, pid_row = (pid_row+1)%np_row) {
		sz_comm_row = (r + sz_blk_row > sz_row) ? sz_row - r : sz_blk_row;
		pid_col = ip_col_start;
		for (int c = 0; c < sz_col; c += sz_blk_col, pid_col = (pid_col+1)%np_col) {
			sz_comm_col = (c + sz_blk_col > sz_col) ? sz_col - c : sz_blk_col;
			if (ip_row == src_row && ip_col == src_col) 
				dgesd2d(&ctxt, &sz_comm_row, &sz_comm_col, A+r+c*sz_row, &sz_row, &pid_row, &pid_col);
			if (ip_row == pid_row && ip_col == pid_col) {
				dgerv2d(&ctxt, &sz_comm_row, &sz_comm_col, A_loc+r_loc+c_loc*sz_loc_row, &sz_loc_row, &src_row, &src_col);
				c_loc = (c_loc + sz_comm_col) % sz_loc_col;
			}
		}
		if (ip_row == pid_row)
			r_loc += sz_comm_row;
	}
}


inline void gather(int const& ctxt, double*& A, double*& A_loc, int const& sz_row, int const& sz_col, int const& sz_blk_row, int const& sz_blk_col, int const& ip_row, int const& ip_col, int const& np_row, int const& np_col, int const& src_row, int const& src_col, int const& ip_row_start, int const& ip_col_start) {
	delete[] A;
	A = new double[sz_row*sz_col];

	int sz_loc_row = ::numroc(&sz_row, &sz_blk_row, &ip_row, &ip_row_start, &np_row);
	int sz_loc_col = ::numroc(&sz_col, &sz_blk_col, &ip_col, &ip_col_start, &np_col);

	int pid_row = ip_row_start, pid_col = ip_col_start;
	int sz_comm_row = 0, sz_comm_col = 0;
	int r_loc = 0, c_loc = 0;
	for (int r = 0; r < sz_row; r += sz_blk_row, pid_row = (pid_row+1)%np_row) {
		sz_comm_row = (r + sz_blk_row > sz_row) ? sz_row - r : sz_blk_row;
		pid_col = 0;
		for (int c = 0; c < sz_col; c += sz_blk_col, pid_col = (pid_col+1)%np_col) {
			sz_comm_col = (c + sz_blk_col > sz_col) ? sz_col - c : sz_blk_col;
			if (ip_row == pid_row && ip_col == pid_col) {
				dgesd2d(&ctxt, &sz_comm_row, &sz_comm_col, A_loc+r_loc+c_loc*sz_loc_row, &sz_loc_row, &src_row, &src_col);
				c_loc = (c_loc + sz_comm_col) % sz_loc_col;
			}
			if (ip_row == src_row && ip_col == src_col)
				dgerv2d(&ctxt, &sz_comm_row, &sz_comm_col, A+r+c*sz_row, &sz_row, &pid_row, &pid_col);
		}
		if (ip_row == pid_row)
			r_loc += sz_comm_row;
	}
}


#endif
