#include <mpi.h>
#include <iostream>
#include <cassert>

extern "C"
{
	// BLACS
	void Cblacs_pinfo(int*, int*);
	void Cblacs_get(int, int, int*);
	void Cblacs_barrier(int, char*);
	void Cblacs_gridinit(int* , char*, int, int);
	void Cblacs_gridinfo(int, int*, int*, int*, int*);
	void Cblacs_gridexit(int);
	void Cdgesd2d(int, int, int, double*, int, int, int);
	void Cdgerv2d(int, int, int, double*, int, int, int);

	// ScaLAPACK utilities
	int numroc_(int const*, int const*, int const*, int const*, int const*);
}

void bcast_int(int root_id, int& data) {
    MPI_Bcast(&data, 1, MPI_INT, root_id, MPI_COMM_WORLD);
}

template <typename ...Ts>
void bcast_int(int root_id, int& data, Ts&... args) {
    MPI_Bcast(&data, 1, MPI_INT, root_id, MPI_COMM_WORLD);
    bcast_int(root_id, args...);
}

void sleep(double i) {
	std::string cmd = "sleep " + std::to_string(i);
	std::system(cmd.c_str());
}

void print(double* mat, int nrow, int ncol, const char* fmt = "%7.2f ") {
    for (int irow = 0; irow < nrow; ++irow) {
        for (int icol = 0; icol < ncol; ++icol) {
            printf(fmt, mat[icol * nrow + irow]); // column major!!!
        }
        std::cout << std::endl;
    }
}

void scatter(
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
		int 	const& 		ip_row_root = 0, // where the global matrix is scattered from
		int 	const& 		ip_col_root = 0, // where the global matrix is scattered from
		int 	const& 		ip_row_start = 0,
		int 	const& 		ip_col_start = 0
) {
	int sz_row_loc = numroc_(&sz_row, &sz_row_blk, &ip_row, &ip_row_start, &np_row);
	int sz_col_loc = numroc_(&sz_col, &sz_col_blk, &ip_col, &ip_col_start, &np_col);

	// receiver's blacs grid indices for the block to communicate
	int ipr = ip_row_start, ipc = ip_col_start;

	// size of the block to communicate
	int szr = 0, szc = 0; 

	// indices in A_loc
	int r_loc = 0, c_loc = 0; 

    // r & c are the indices of the upper-left element element of the block in the global matrix to be communicated
	for (int r = 0; r < sz_row; r += sz_row_blk, ipr = (ipr+1)%np_row) {

        // row size of the block to be communicated
		szr = (r + sz_row_blk > sz_row) ? sz_row - r : sz_row_blk;

		for (int c = 0; c < sz_col; c += sz_col_blk, ipc = (ipc+1)%np_col) {

            // column size of the block to be communicated
			szc = (c + sz_col_blk > sz_col) ? sz_col - c : sz_col_blk;

            // send
			if (ip_row == ip_row_root && ip_col == ip_col_root) {
				Cdgesd2d(ctxt, szr, szc, A+r+c*sz_row, sz_row, ipr, ipc);
            }

            // receive
			if (ip_row == ipr && ip_col == ipc) {
				Cdgerv2d(ctxt, szr, szc, A_loc+r_loc+c_loc*sz_row_loc, sz_row_loc, ip_row_root, ip_col_root);
				c_loc = (c_loc + szc) % sz_col_loc;
			}
		}

		ipc = ip_col_start;
		if (ip_row == ipr)
			r_loc += szr;
	}
}

void gather(
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
	int sz_row_loc = numroc_(&sz_row, &sz_row_blk, &ip_row, &ip_row_start, &np_row);
	int sz_col_loc = numroc_(&sz_col, &sz_col_blk, &ip_col, &ip_col_start, &np_col);

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
				Cdgesd2d(ctxt, szr, szc, A_loc+r_loc+c_loc*sz_row_loc, sz_row_loc, ip_row_root, ip_col_root);
				c_loc = (c_loc + szc) % sz_col_loc;
			}
			if (ip_row == ip_row_root && ip_col == ip_col_root)
				Cdgerv2d(ctxt, szr, szc, A+r+c*sz_row, sz_row, ipr, ipc);
		}
		ipc = ip_col_start;
		if (ip_row == ipr)
			r_loc += szr;
	}
}

int main() {

	MPI_Init(nullptr, nullptr);

	int iZERO = 0;

    int ip_src_row = 0;
    int ip_src_col = 0;

    // initialize MPI proc info
	int id, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // initialize BLACS proc & context info
	int ctxt, id_blacs, np_blacs;
	Cblacs_pinfo(&id_blacs, &np_blacs);
	Cblacs_get(iZERO, iZERO, &ctxt);
	
    // read in parameters
	int np_row = 0, np_col = 0;
	int nrow = 0, ncol = 0;
	int nrow_blk = 0, ncol_blk = 0;

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
			std::cout << "mpi id = " << id << ", " 
				<< "blacs id = " << id_blacs << std::endl;
		}
		sleep(0.2);
	}

	if (id == 0) {
        // processer grid
		std::cout << "enter the number of rows for the processer grid:" << std::endl;
		std::cin >> np_row;
		std::cout << "enter the number of columns for the processer grid:" << std::endl;
		std::cin >> np_col;

        assert(np_row * np_col == nprocs && "total number of processors does not match the grid size");

        // global matrix size
		std::cout << "nrow:" << std::endl;
		std::cin >> nrow;
		std::cout << "ncol:" << std::endl;
		std::cin >> ncol;

        // block size
		std::cout << "nrow_blk:" << std::endl;
		std::cin >> nrow_blk;
		std::cout << "ncol_blk:" << std::endl;
		std::cin >> ncol_blk;

        // blacs grid coordinate where the first block is distributed
		std::cout << "ip_src_row:" << std::endl;
		std::cin >> ip_src_row;
		std::cout << "ip_src_col:" << std::endl;
		std::cin >> ip_src_col;

		std::cout << std::endl;
	}

    // let all procs know the parameters
    bcast_int(0, np_row, np_col, nrow, ncol, nrow_blk, ncol_blk, ip_src_row, ip_src_col);

	int ip_row, ip_col;
	char layout = 'C';

    // initialize BLACS grid
	Cblacs_gridinit(&ctxt, &layout, np_row, np_col);
	Cblacs_gridinfo(ctxt, &np_row, &np_col, &ip_row, &ip_col);

    double* A = nullptr; // global matrix (only allocated in proc 0)
    double* A_loc = nullptr; // local matrix

	if (id == 0) {
        A = new double[nrow * ncol];
        for (int r = 0; r < nrow; ++r) {
            for (int c = 0; c < ncol; ++c) {
                A[r + c * nrow] = r + 0.01 * c; // column major!!!
            }
        }

        std::cout << "A = " << std::endl;
        print(A, nrow, ncol);
        std::cout << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
    // number of local rows and columns after block-cyclic distribution
	int nrow_loc = numroc_(&nrow, &nrow_blk, &ip_row, &ip_src_row, &np_row);
	int ncol_loc = numroc_(&ncol, &ncol_blk, &ip_col, &ip_src_col, &np_col);

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
            printf("grid_id = (%i, %i), loc size = (%i, %i)\n", ip_row, ip_col, nrow_loc, ncol_loc);
		}
		sleep(0.2);
	}

    A_loc = new double[nrow_loc * ncol_loc];

	scatter(ctxt, A, A_loc, nrow, ncol, nrow_blk, ncol_blk, ip_row, ip_col, np_row, np_col, 0, 0, ip_src_row, ip_src_col);

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
            printf("mpi_id = %i   grid_id = (%i, %i)\n", id, ip_row, ip_col);
            std::cout << "A_loc = " << std::endl;
            print(A_loc, nrow_loc, ncol_loc);
            std::cout << std::endl;
		}
		sleep(0.2);
	}

    // rescale A_loc
    for (int i = 0; i < nrow_loc * ncol_loc; ++i) {
        A_loc[i] *= -1;
    }

    // gather scaled A_loc to B
    double* B = nullptr;
	if (id == 0) {
		B = new double[nrow * ncol];
	}

	gather(ctxt, B, A_loc, nrow, ncol, nrow_blk, ncol_blk, ip_row, ip_col, np_row, np_col, 0, 0, ip_src_row, ip_src_col);

	if (id == 0) {
		std::cout << "B = " << std::endl;
		print(B, nrow, ncol);
	}

    // clean up 
    delete[] A;
    delete[] A_loc;
    delete[] B;

	Cblacs_gridexit(ctxt);
	MPI_Finalize();

	return 0;
}
