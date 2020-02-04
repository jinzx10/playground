#include <mpi.h>
#include <armadillo>
#include <sstream>

using namespace arma;

void gather() {
}

void gather(vec const& local, mat& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_DOUBLE, global.memptr(), local.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void gather(uvec const& local, umat& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, global.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

void gather(vec const& local, vec& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_DOUBLE, global.memptr(), local.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void gather(uvec const& local, uvec& global) {
	::MPI_Gather(local.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, global.memptr(), local.n_elem, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

template <typename eT, typename ...Ts>
void gather(arma::Col<eT> const& local, arma::Mat<eT>& global, Ts& ...args) {
	gather(local, global);
	gather(args...);
}

template <typename eT, typename ...Ts>
void gather(arma::Col<eT> const& local, arma::Col<eT>& global, Ts& ...args) {
	gather(local, global);
	gather(args...);
}


int main(int argc, char**argv) {
	int num_procs;
	int id;

	::MPI_Init(nullptr, nullptr);
	::MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	::MPI_Comm_rank(MPI_COMM_WORLD, &id);

	uword sz;

	if (id == 0) {
		std::stringstream ss;
		ss << argv[1];
		ss >> sz;
	}

	::MPI_Bcast(&sz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

	mat m1, m2;
	umat m3;
	vec v1, v2;
	uvec v3;

	vec v_local1 = id * ones(sz);
	vec v_local2 = -id * ones(sz);
	uvec v_local3 = 2*id * ones<uvec>(sz);

	if (id == 0) {
		m1.zeros(sz, num_procs);
		m2.zeros(sz, num_procs);
		m3.zeros(sz, num_procs);
		v1.zeros(sz*num_procs);
		v2.zeros(sz*num_procs);
		v3.zeros(sz*num_procs);
	}

	//::MPI_Gather(v_local.memptr(), sz, MPI_DOUBLE, m.memptr(), sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	gather(v_local1, m1, v_local2, m2, v_local3, m3);
	gather(v_local1, v1, v_local2, v2, v_local3, v3);

	if (id == 0) {
		m1.print();
		std::cout << std::endl;
		m2.print();
		std::cout << std::endl;
		m3.print();
		std::cout << std::endl;
		std::cout << std::endl;
		v1.print();
		std::cout << std::endl;
		v2.print();
		std::cout << std::endl;
		v3.print();
	}

	::MPI_Finalize();
	return 0;
}
