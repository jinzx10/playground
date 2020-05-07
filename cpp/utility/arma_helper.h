#ifndef __ARMA_HELPER_H__
#define __ARMA_HELPER_H__

#include <armadillo>
#include <initializer_list>

// broadcasted plus/minus between column and row vectors
template <typename R, typename C>
typename std::enable_if< R::is_row && C::is_col, arma::Mat<decltype(std::declval<typename C::elem_type>()+std::declval<typename R::elem_type>())> >::type bcast_plus(C const& col, R const& row) {
	return arma::repmat(col, 1, arma::size(row).n_cols).eval().each_row() + row;
}

template <typename R, typename C>
typename std::enable_if< R::is_row && C::is_col, arma::Mat<decltype(std::declval<typename R::elem_type>()+std::declval<typename C::elem_type>())> >::type bcast_plus(R const& row, C const& col) {
	return arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col() + col;
}

template <typename R, typename C>
typename std::enable_if< R::is_row && C::is_col, arma::Mat<decltype(std::declval<typename C::elem_type>()-std::declval<typename R::elem_type>())> >::type bcast_minus(C const& col, R const& row) {
	return arma::repmat(col, 1, arma::size(row).n_cols).eval().each_row() - row;
}

template <typename R, typename C>
typename std::enable_if< R::is_row && C::is_col, arma::Mat<decltype(std::declval<typename R::elem_type>()-std::declval<typename C::elem_type>())> >::type bcast_minus(R const& row, C const& col) {
	return arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col() - col;
}


// unit vector
inline arma::vec unit_vec(arma::uword const& dim, arma::uword const& index) {
	arma::vec v = arma::zeros(dim);
	v(index) = 1.0;
	return v;
}


// index range and concatenation
inline arma::uvec range(arma::uword const& i, arma::uword const& j) {
	return arma::regspace<arma::uvec>(i, 1, j); // end-inclusive
}

template <typename T>
arma::uvec cat(T const& i) {
	return arma::uvec{i};
}

template <typename T, typename ...Ts>
arma::uvec cat(T const& i, Ts const& ...args) {
	return arma::join_cols(arma::uvec{i}, cat(args...));
}


// mass size setting
template <typename eT>
void set_size(arma::uword const& sz, arma::Mat<eT>& m) {
	m.set_size(sz);
}

template <typename eT, typename ...Ts>
void set_size(arma::uword const& sz, arma::Mat<eT>& m, Ts& ...args) {
    m.set_size(sz);
    set_size(sz, args...);
}

template <typename eT>
void set_size(arma::uword const& sz_r, arma::uword const& sz_c, arma::Mat<eT>& m) {
	m.set_size(sz_r, sz_c);
}

template <typename eT, typename ...Ts>
void set_size(arma::uword const& sz_r, arma::uword const& sz_c, arma::Mat<eT>& m, Ts& ...args) {
    m.set_size(sz_r, sz_c);
    set_size(sz_r, sz_c, args...);
}



// matrix concatenation
template <typename T>
T join_r(std::initializer_list<T> m) {
	T z;
	for (auto it = m.begin(); it != m.end(); ++it) {
		z = join_rows(z, *it);
	}
	return z;
}

template <typename T>
T join(std::initializer_list< std::initializer_list<T> > m) {
	T z;
	for (auto it = m.begin(); it != m.end(); ++it) {
		z = join_cols(z, join_r(*it));
	}
	return z;
}

template <typename T>
T join_r(T const& m1, T const& m2) {
	return arma::join_rows(m1, m2);
}

template <typename T, typename ...Ts>
T join_r(T const& m, Ts const& ...ms) {
    return join_rows( m, join_r(ms...) );
}

template <typename T>
T join_c(T const& m1, T const& m2) {
	return arma::join_cols(m1, m2);
}

template <typename T, typename ...Ts>
T join_c(T const& m, Ts const& ...ms) {
    return join_cols( m, join_c(ms...) );
}

template <typename eT>
arma::Mat<eT> join_d(arma::Mat<eT> const& m1, arma::Mat<eT> const& m2) {
    return join_cols(
			join_rows( m1, arma::zeros<arma::Mat<eT>>(m1.n_rows, m2.n_cols) ),
			join_rows( arma::zeros<arma::Mat<eT>>(m2.n_rows, m1.n_cols), m2 )
	);
}

template <typename T, typename ...Ts>
T join_d(T const& m, Ts const& ...ms) {
    return join_d(m, join_d(ms...));
}


// save/load
template <arma::file_type F, typename T>
void arma_save(std::string const& dir, T const& data, std::string const& name) {
    data.save(dir+"/"+name, F); 
}

template <arma::file_type F, typename T, typename ...Ts>
void arma_save(std::string const& dir, T const& data, std::string const& name, Ts const& ...args) {
    data.save(dir+"/"+name, F); 
    arma_save<F>(dir, args...);
}

template <typename T>
void arma_load(std::string const& dir, T& data, std::string const& name) {
    data.load(dir+"/"+name);
}

template <typename T, typename ...Ts>
void arma_load(std::string const& dir, T& data, std::string const& name, Ts& ...args) {
    data.load(dir+"/"+name);
    arma_load(dir, args...);
}



#endif
