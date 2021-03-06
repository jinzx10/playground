/* This test program reads two matrices from file, scatters it to different processes,
 * calls pdgemm to perform a parallel matrix multiplication, and gathers the result.
 *
 * On some machines, if scalapack is compiled with user-specified blas and lapack libs
 * (like mkl or openblas), the link line should include those libs (like -lopenblas or
 * -lmkl_core -lmkl_...) as well as fortran libs (-lgfortran or -lifcore), if blas and
 * lapack routines (like pdgemm) are called.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <mpi.h>
#include "../fstream/matio.h"
#include "scalapack.h"
#include "aux.h"
#include <armadillo>

int main(int argc, char** argv) {

	MPI_Init(nullptr, nullptr);

	int ZERO = 0, ONE = 1;
	double dZERO = 0.0, dONE = 1.0;
	int ctxt = 0;
	int id = 0, nprocs = 0;

	Cblacs_pinfo(&id, &nprocs);
	Cblacs_get(0, 0, &ctxt);

	// command line input check
	if (argc < 7) {
		if (!id)
			std::cerr << "Usage: mpirun -np X ./mkl_pdgemm data1.txt data2.txt P Q mb nb" << std::endl;
		MPI_Finalize();
		return -1;
	}

	int np_row = 0, np_col = 0;
	int ip_row = 0, ip_col = 0;
	char layout[] = "Row";

	if (!id) {
		std::stringstream ss;
		ss << argv[3] << ' ' << argv[4];
		ss >> np_row >> np_col;
	}
	MPI_Bcast(&np_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&np_col, 1, MPI_INT, 0, MPI_COMM_WORLD);

	Cblacs_gridinit(&ctxt, layout, np_row, np_col);
	Cblacs_gridinfo(ctxt, &np_row, &np_col, &ip_row, &ip_col);

	double* A = nullptr;
	double* B = nullptr;
	double* C = nullptr;
	int MA, NA, MB, NB;
	std::string file1, file2;
	if (!id) {
		std::stringstream ss;
		ss << argv[1] << ' ' << argv[2];
		ss >> file1 >> file2;
		read_mat(file1, A, MA, NA);
		read_mat(file2, B, MB, NB);
	}
	MPI_Bcast(&MA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&NA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&MB, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&NB, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if (NA != MB) {
		if (!id)
			std::cerr << "invalid size for matrix multiplication." << std::endl;
		MPI_Finalize();
		return -1;
	}

	int mb, nb;
	if (!id) {
		std::stringstream ss;
		ss << argv[5] << ' ' << argv[6];
		ss >> mb >> nb;
	}
	MPI_Bcast(&mb, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int RA = numroc_(&MA, &mb, &ip_row, &ZERO, &np_row);
	int CA = numroc_(&NA, &nb, &ip_col, &ZERO, &np_col);
	int RB = numroc_(&MB, &mb, &ip_row, &ZERO, &np_row);
	int CB = numroc_(&NB, &nb, &ip_col, &ZERO, &np_col);
	int RC = numroc_(&MA, &mb, &ip_row, &ZERO, &np_row);
	int CC = numroc_(&NB, &nb, &ip_col, &ZERO, &np_col);

	int descA[9], descB[9], descC[9];

	int info;
	descinit_(descA, &MA, &NA, &mb, &nb, &ZERO, &ZERO, &ctxt, &RA, &info);
	descinit_(descB, &MB, &NB, &mb, &nb, &ZERO, &ZERO, &ctxt, &RB, &info);
	descinit_(descC, &MA, &NB, &mb, &nb, &ZERO, &ZERO, &ctxt, &RC, &info);

	// scatter matrix
	double* A_loc = nullptr;
	double* B_loc = nullptr;
	double* C_loc = nullptr;

	scatter(ctxt, A, A_loc, MA, NA, mb, nb, ip_row, ip_col, np_row, np_col);
	scatter(ctxt, B, B_loc, MB, NB, mb, nb, ip_row, ip_col, np_row, np_col); 

	C_loc = new double[RC*CC];
	for (int i = 0; i != RC*CC; ++i) C_loc[i] = 0.0;

	char trans = 'N';

	// ia, ja, ... start from one instead of zero! (fortran convention)
	pdgemm_(&trans, &trans, &MA, &NB, &NA, &dONE, A_loc, &ONE, &ONE, descA, B_loc, &ONE, &ONE, descB, &dZERO, C_loc, &ONE, &ONE, descC);

	gather(ctxt, C, C_loc, MA, NB, mb, nb, ip_row, ip_col, np_row, np_col);

	if (!id) {
		print_mat(C, MA, NB);
		arma::mat a, b, c;
		a.load(file1);
		b.load(file2);
		c = a*b;
		c.print();
	}

	delete[] A;
	delete[] A_loc;
	delete[] B;
	delete[] B_loc;
	delete[] C;
	delete[] C_loc;

	Cblacs_gridexit(ctxt);
	MPI_Finalize();

	return 0;
}

