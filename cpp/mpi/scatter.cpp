#include "helper.h"
#include <armadillo>
#include "../utility/widgets.h"
#include "../utility/mpi_helper.h"

using namespace arma;

void sleep(double i) {
	std::string cmd = "sleep " + std::to_string(i);
	std::system(cmd.c_str());
}

int main(int, char** argv) {

	MPI_Init(nullptr, nullptr);

	int iZERO = 0, iONE = 1;
	double dZERO = 0.0, dONE = 1.0;

	int id, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int ctxt, id_blacs, np_blacs;
	blacs_pinfo(&id_blacs, &np_blacs);
	blacs_get(&iZERO, &iZERO, &ctxt);
	
	int np_row = 0, np_col = 0;
	int szA_row = 0, szA_col = 0;
	int szA_row_blk = 0, szA_col_blk = 0;

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
			std::cout << "mpi id = " << id << ", " 
				<< "blacs id = " << id_blacs << std::endl;
		}
		sleep(0.2);
	}

	if (id == 0) {
		std::cout << "np_row:" << std::endl;
		std::cin >> np_row;
		std::cout << "np_col:" << std::endl;
		std::cin >> np_col;
		std::cout << "szA_row:" << std::endl;
		std::cin >> szA_row;
		std::cout << "szA_col:" << std::endl;
		std::cin >> szA_col;
		std::cout << "szA_row_blk:" << std::endl;
		std::cin >> szA_row_blk;
		std::cout << "szA_col_blk:" << std::endl;
		std::cin >> szA_col_blk;
		std::cout << std::endl;
		//readargs(argv, np_row, np_col, szA_row, szA_col, szA_row_blk, szA_col_blk);
	}
	bcast(np_row, np_col, szA_row, szA_col, szA_row_blk, szA_col_blk);

	int ip_row, ip_col;
	char layout = 'C';

	blacs_gridinit(&ctxt, &layout, &np_row, &np_col);
	blacs_gridinfo(&ctxt, &np_row, &np_col, &ip_row, &ip_col);

	mat A, A_loc;


	if (id == 0) {
		A.set_size(szA_row, szA_col);
		A = 0.01 * ones(szA_row) * regspace<rowvec>(0, 1, szA_col-1) + regspace(0, 1, szA_row-1) * ones<rowvec>(szA_col);
		std::cout << "A" << std::endl;
		A.print();
		std::cout << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	sleep(0.2);
	
	int szA_row_loc = numroc(&szA_row, &szA_row_blk, &ip_row, &iZERO, &np_row);
	int szA_col_loc = numroc(&szA_col, &szA_col_blk, &ip_col, &iZERO, &np_col);

	A_loc.set_size(szA_row_loc, szA_col_loc);

	scatter(ctxt, A.memptr(), A_loc.memptr(), szA_row, szA_col, szA_row_blk, szA_col_blk, ip_row, ip_col, np_row, np_col);

	for (int i = 0; i != nprocs; ++i) {
		if (id == i) {
			std::cout << "mpi id = " << id << ", grid id = ("
				<< ip_row << "," << ip_col << ")" << std::endl
				<< "A_loc = " << std::endl;
			A_loc.print();
			std::cout << std::endl;
		}
		sleep(0.2);
	}

	A_loc *= 0.1;

	mat B;
	if (id == 0) {
		B.set_size(size(A));
	}

	gather(ctxt, B.memptr(), A_loc.memptr(), szA_row, szA_col, szA_row_blk, szA_col_blk, ip_row, ip_col, np_row, np_col);

	if (id == 0) {
		std::cout << "B = " << std::endl;
		B.print();
	}


	blacs_gridexit(&ctxt);
	MPI_Finalize();
	return 0;
}
