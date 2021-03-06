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

template <typename T1, typename T2, char op>
struct binary_return_t
{};

template <typename T1, typename T2>
struct binary_return_t<T1, T2, '+'>
{
	using return_t = decltype(std::declval<T1>() + std::declval<T2>());
};

template <typename T1, typename T2>
struct binary_return_t<T1, T2, '-'>
{
	using return_t = decltype(std::declval<T1>() - std::declval<T2>());
};

template <typename T1, typename T2>
struct binary_return_t<T1, T2, '*'>
{
	using return_t = decltype(std::declval<T1>() * std::declval<T2>());
};

template <typename T1, typename T2>
struct binary_return_t<T1, T2, '/'>
{
	using return_t = decltype(std::declval<T1>() / std::declval<T2>());
};

template <char op, typename R, typename C>
typename std::enable_if< R::is_row && C::is_col, arma::Mat<typename binary_return_t<typename C::elem_type, typename R::elem_type, op>::return_t> >::type bcast_op(C const& col, R const& row) {
	switch (op) {
		case '+':
			return col + arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col();
		case '-':
			return col - arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col();
		case '*':
			return col * row;
		case '/':
			return col / arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col();
		default:
			return {};
	}
}

template <char op, typename R, typename C>
typename std::enable_if< R::is_row && C::is_col, arma::Mat<typename binary_return_t<typename R::elem_type, typename C::elem_type, op>::return_t> >::type bcast_op(R const& row, C const& col) {
	switch (op) {
		case '+':
			return arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col() + col;
		case '-':
			return arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col() - col;
		case '*':
			return col * row;
		case '/':
			return arma::repmat(row, arma::size(col).n_rows, 1).eval().each_col() / col;
		default:
			return {};
	}
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

// dimensions of armadillo Col/Row/Mat/Cube
template <typename eT>
arma::uvec dim(arma::Col<eT> const& a) {
	return arma::uvec{a.n_elem};
}

template <typename eT>
arma::uvec dim(arma::Row<eT> const& a) {
	return arma::uvec{a.n_elem};
}

template <typename eT>
arma::uvec dim(arma::Mat<eT> const& a) {
	return arma::uvec{a.n_rows, a.n_cols};
}

template <typename eT>
arma::uvec dim(arma::Cube<eT> const& a) {
	return arma::uvec{a.n_rows, a.n_cols, a.n_slices};
}

// batch size setting
template <typename eT>
int set_size(arma::uvec const& sz, arma::Row<eT>& a) {
	return sz.n_elem == 1 ? a.set_size(sz(0)), 0 : 1;
}

template <typename eT>
int set_size(arma::uvec const& sz, arma::Col<eT>& a) {
	return sz.n_elem == 1 ? a.set_size(sz(0)), 0 : 1;
}

template <typename eT>
int set_size(arma::uvec const& sz, arma::Mat<eT>& a) {
	return sz.n_elem == 2 ? a.set_size(sz(0), sz(1)), 0 : 1;
}

template <typename eT>
int set_size(arma::uvec const& sz, arma::Cube<eT>& a) {
	return sz.n_elem == 3 ? a.set_size(sz(0), sz(1), sz(2)), 0 : 1;
}

template <typename T, typename ...Ts>
int set_size(arma::uvec const& sz, T& a, Ts& ...args) {
	int status = set_size(sz, a);
	return status ? status : set_size(sz, args...);
}

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
	for (auto it = m.begin(); it != m.end(); ++it)
		z = join_rows(z, *it);
	return z;
}

template <typename T>
T join(std::initializer_list< std::initializer_list<T> > m) {
	T z;
	for (auto it = m.begin(); it != m.end(); ++it)
		z = join_cols(z, join_r(*it));
	return z;
}

template <typename T1, typename T2>
auto join_r(T1 const& m1, T2 const& m2) {
	return arma::join_rows(m1, m2);
}

template <typename T, typename ...Ts>
auto join_r(T const& m, Ts const& ...ms) {
    return arma::join_rows(m, join_r(ms...)).eval(); // slow, but it needs eval() to work, why?
}

template <typename T1, typename T2>
auto join_c(T1 const& m1, T2 const& m2) {
	return arma::join_cols(m1, m2);
}

template <typename T, typename ...Ts>
auto join_c(T const& m, Ts const& ...ms) {
    return arma::join_cols(m, join_c(ms...)).eval();
}

template <typename T1, typename T2>
auto join_d(T1 const& m1, T2 const& m2) {
    return join_cols(
			join_rows( m1, arma::zeros<arma::Mat<typename T1::elem_type>>(m1.n_rows, m2.n_cols) ),
			join_rows( arma::zeros<arma::Mat<typename T2::elem_type>>(m2.n_rows, m1.n_cols), m2 )
	).eval();
}

template <typename T, typename ...Ts>
auto join_d(T const& m, Ts const& ...ms) {
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
