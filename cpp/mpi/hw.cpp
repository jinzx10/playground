#include <iostream>
#include <mpi.h>
#include <armadillo>
#include "mpiaux.h"

int main() {

	int id = 0, nprocs = 0;

	mpi_start(&id, &nprocs);
	print_id(&id, &nprocs);

	int sz = 3;

	/* built-in array */
	double* a = new double[sz];
	for (int i = 0; i != sz; ++i) a[i] = id;

	double* A = nullptr;
	if (!id) {
		A = new double[nprocs*sz];
		for (int i = 0; i != nprocs*sz; ++i) A[i] = 0;
	}

	::MPI_Gather(a, sz, MPI_DOUBLE, A, sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (!id) print_mat(A, nprocs, sz, false);

	delete [] a;
	if (!id) delete [] A;

	/* armadillo */
	arma::vec v = arma::linspace<arma::vec>(0, sz-1, sz) * id;
	arma::vec u = arma::zeros(sz);
	arma::mat m;
	if (!id) m = arma::zeros(sz-1, nprocs);

	::MPI_Gather(v.begin_row(1), sz-1, MPI_DOUBLE, m.memptr(), sz-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	::MPI_Reduce(v.memptr(), u.memptr(), sz, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (!id) {
		m.print();
		u.print();
	}

	mpi_end();

	return 0;
}
