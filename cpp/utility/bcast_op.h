#ifndef __BROADCASTED_OPERATIONS_H__
#define __BROADCASTED_OPERATIONS_H__

#include <type_traits>
#include <armadillo>

template <typename T1, typename T2, typename Op>
using return_t = decltype( std::declval<Op>()( std::declval<T1>(), std::declval<T2>() ) );

template <typename T1, typename T2, typename Op>
typename std::enable_if< ( std::declval<Op>()(std::declval<T1>(), std::declval<typename T2::elem_type>())  , T1::is_col && T2::is_row), arma::Mat< return_t<typename T1::elem_type, typename T2::elem_type, Op> > >::type bcast_op(T1 const& col, T2 const& row, Op op) {
	typename std::enable_if<T1::is_col && T2::is_row, arma::Mat< return_t< typename T1::elem_type, typename T2::elem_type, Op> > >::type result(arma::size(col).n_rows, arma::size(row).n_cols);
	for (arma::uword j = 0; j != arma::size(row).n_cols; ++j) {
		result.col(j) = op(col, arma::conv_to<arma::Row<typename T2::elem_type>>::from(row)(j));
	}
	return result;
}

template <typename T1, typename T2, typename Op>
typename std::enable_if< T1::is_row && T2::is_col, arma::Mat< return_t<typename T1::elem_type, typename T2::elem_type, Op> > >::type bcast_op(T1 const& row, T2 const& col, Op op) {
	typename std::enable_if<T1::is_row && T2::is_col, arma::Mat< return_t< typename T1::elem_type, typename T2::elem_type, Op> > >::type result(arma::size(col).n_rows, arma::size(row).n_cols);
	for (arma::uword j = 0; j != arma::size(row).n_cols; ++j) {
		result.col(j) = op(arma::conv_to<arma::Row<typename T2::elem_type>>::from(row)(j), col);
	}
	return result;
}

#endif
