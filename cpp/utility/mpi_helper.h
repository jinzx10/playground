#ifndef __MPI_HELPER_H__
#define __MPI_HELPER_H__

#include <mpi.h>
#include <iostream>
#include <type_traits>
#include <armadillo>

template <typename eT>
MPI_Datatype mpi_type_helper() {
	std::cerr << "fails to convert to MPI datatype" << std::endl;
	return 0;
}

template<>
inline MPI_Datatype mpi_type_helper<char>() {
	return MPI_CHAR;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned char>() {
	return MPI_UNSIGNED_CHAR;
}

template<>
inline MPI_Datatype mpi_type_helper<float>() {
	return MPI_FLOAT;
}

template<>
inline MPI_Datatype mpi_type_helper<double>() {
	return MPI_DOUBLE;
}

template<>
inline MPI_Datatype mpi_type_helper<long double>() {
	return MPI_LONG_DOUBLE;
}

template<>
inline MPI_Datatype mpi_type_helper<short>() {
	return MPI_SHORT;
}

template<>
inline MPI_Datatype mpi_type_helper<int>() {
	return MPI_INT;
}

template<>
inline MPI_Datatype mpi_type_helper<long>() {
	return MPI_LONG;
}

template<>
inline MPI_Datatype mpi_type_helper<long long>() {
	return MPI_LONG_LONG;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned short>() {
	return MPI_UNSIGNED_SHORT;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned int>() {
	return MPI_UNSIGNED;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned long>() {
	return MPI_UNSIGNED_LONG;
}

template<>
inline MPI_Datatype mpi_type_helper<unsigned long long>() {
	return MPI_UNSIGNED_LONG_LONG;
}


// broadcast
template <typename T>
typename std::enable_if<std::is_trivial<T>::value, int>::type bcast(T& data) {
	return MPI_Bcast(&data, 1, mpi_type_helper<T>(), 0, MPI_COMM_WORLD);
}

template <typename T>
typename std::enable_if<arma::is_arma_type<T>::value, int>::type bcast(T& data) {
	return MPI_Bcast(data.memptr(), data.n_elem, mpi_type_helper<typename T::elem_type>(), 0, MPI_COMM_WORLD);
}

template <typename T, typename ...Ts>
int bcast(T& data, Ts& ...args) {
	int status = bcast(data);
	if (status)
		return status;
	return bcast(args...);
}


// gather and gatherv
// use "gather" if the number of elements to gather is the same for every process
// use "gatherv" otherwise
template <typename eT>
int gather(eT const& local, arma::Mat<eT>& global) {
	return MPI_Gather(&local, 1, mpi_type_helper<eT>(), global.memptr(), 1, mpi_type_helper<eT>(), 0, MPI_COMM_WORLD);
}

template <typename eT>
int gather(arma::Mat<eT> const& local, arma::Mat<eT>& global) {
	return MPI_Gather(local.memptr(), local.n_elem, mpi_type_helper<eT>(), global.memptr(), local.n_elem, mpi_type_helper<eT>(), 0, MPI_COMM_WORLD);
}

template <typename eT, typename ...Ts>
int gather(eT const& local, arma::Mat<eT>& global, Ts& ...args) {
	int status = gather(local, global);
	if (status)
		return status;
	return gather(args...);
}

template <typename eT, typename ...Ts>
int gather(arma::Mat<eT> const& local, arma::Mat<eT>& global, Ts& ...args) {
	int status = gather(local, global);
	if (status)
		return status;
	return gather(args...);
}

template <typename eT>
int gatherv(arma::Mat<eT> const& local, arma::Mat<eT>& global, int root = 0) {
	int id, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	int n_local = local.n_elem;
	arma::Col<int> local_counts, disp;
   	if (id == root) 
		local_counts = arma::zeros<arma::Col<int>>(nprocs);
	gather(n_local, local_counts);
	if (id == root) {
		disp = arma::cumsum(local_counts);
		disp.insert_rows(0,1,true);
		disp.shed_row(disp.n_elem-1);
	}
	return MPI_Gatherv(local.memptr(), local.n_elem, mpi_type_helper<eT>(), global.memptr(), local_counts.memptr(), disp.memptr(), mpi_type_helper<eT>(), root, MPI_COMM_WORLD);
}

template <typename eT, typename ...Ts>
int gatherv(arma::Mat<eT> const& local, arma::Mat<eT>& global, Ts& ...args) {
	int status = gatherv(local, global);
	if (status) {
		return status;
	}
	return gatherv(args...);
}


#endif