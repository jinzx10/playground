#include <mpi.h>
#include <armadillo>
#include <sstream>

using namespace arma;

template <typename eT>
inline MPI_Datatype get_type() {
	return MPI_DOUBLE;
}

template <>
inline MPI_Datatype get_type<char>() {
	return MPI_CHAR;
}

template <>
inline MPI_Datatype get_type<double>() {
	return MPI_DOUBLE;
}

template <>
inline MPI_Datatype get_type<int>() {
	return MPI_INT;
}

template <>
inline MPI_Datatype get_type<unsigned long long>() {
	return MPI_UNSIGNED_LONG_LONG;
}

//void gather() {
//}

template <typename eT>
void gather(arma::Mat<eT> const& local, arma::Mat<eT>& global) {
	::MPI_Gather(local.memptr(), local.n_elem, get_type<eT>(), global.memptr(), local.n_elem, get_type<eT>(), 0, MPI_COMM_WORLD);
}

//template <typename eT>
//void gather(arma::Col<eT> const& local, arma::Col<eT>& global) {
//	::MPI_Gather(local.memptr(), local.n_elem, get_type<eT>(), global.memptr(), local.n_elem, get_type<eT>(), 0, MPI_COMM_WORLD);
//}

template <typename eT, typename ...Ts>
void gather(arma::Mat<eT> const& local, arma::Mat<eT>& global, Ts& ...args) {
	gather(local, global);
	gather(args...);
}

//template <typename eT, typename ...Ts>
//void gather(arma::Col<eT> const& local, arma::Col<eT>& global, Ts& ...args) {
//	gather(local, global);
//	gather(args...);
//}

int main(int, char**argv) {
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

	mat dm;
	Mat<int> im;
	umat um;
	vec dv;
	Col<int> iv;
	uvec uv;

	vec dv_local = id * ones(sz);
	Col<int> iv_local = -id * ones<Col<int>>(sz);
	uvec uv_local = 2*id * ones<uvec>(sz);

	if (id == 0) {
		dm.zeros(sz, num_procs);
		im.zeros(sz, num_procs);
		um.zeros(sz, num_procs);
		dv.zeros(sz*num_procs);
		iv.zeros(sz*num_procs);
		uv.zeros(sz*num_procs);
	}

	gather(dv_local, dm, iv_local, im, uv_local, um);
	gather(dv_local, dv, iv_local, iv, uv_local, uv);

	if (id == 0) {
		dm.print();
		std::cout << std::endl;
		im.print();
		std::cout << std::endl;
		um.print();
		std::cout << std::endl;
		std::cout << std::endl;
		dv.print();
		std::cout << std::endl;
		iv.print();
		std::cout << std::endl;
		uv.print();
	}

	std::cout << MPI_COMM_WORLD << std::endl;
	std::cout << typeid(MPI_COMM_WORLD).name() << std::endl;

	::MPI_Finalize();
	return 0;
}

