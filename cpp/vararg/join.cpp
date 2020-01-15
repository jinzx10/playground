#include <armadillo>
#include <initializer_list>

using namespace arma;

template <typename T>
T join_r( std::initializer_list<T> m) {
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

/*
mat joinr2(std::initializer_list<mat> T);

template <typename eT>
arma::Mat<eT> joinr( std::initializer_list<arma::Mat<eT> > m) {
	arma::Mat<eT> z;
	for (auto it = m.begin(); it != m.end(); ++it) {
		//z.insert_cols(z.n_cols, *it);
		z = join_rows(z, *it);
	}
	return z;
}

template <typename eT>
arma::Mat<eT> join(std::initializer_list< std::initializer_list<arma::Mat<eT> > > m) {
	arma::Mat<eT>z;
	for (auto it = m.begin(); it != m.end(); ++it) {
		//z.insert_rows(z.n_rows, joinr(*it));
		z = join_cols(z, joinr(*it));
	}
	return z;
}
*/
/*
mat joinr(mat const& m1, mat const& m2) {
	return join_rows(m1, m2);
}

template <typename ...Ts>
mat joinr(mat const& m, Ts const& ...ms) {
	return join_rows( m, joinr(ms...) );
}

template <typename ...Ts>
mat join(std::initializer_list<mat> m, Ts ...ms) {
	return join_cols( joinr2(m), join(ms...) );
}

mat join(std::initializer_list<mat> m1, std::initializer_list<mat> m2) {
	return join_cols( joinr2(m1), joinr2(m2) );
}

mat joinr2(std::initializer_list<mat> T) {
	mat z;
	for (auto it = T.begin(); it != T.end(); ++it) {
		z.insert_cols(z.n_cols, *it);
	}
	return z;
}

mat join2(std::initializer_list< std::initializer_list<mat> > T) {
	mat z;
	for (auto it = T.begin(); it != T.end(); ++it) {
		z.insert_rows(z.n_rows, joinr2(*it));
	}
	return z;
}
*/

int main() {
	uword sz = 3;
	mat m1 = zeros(sz, 1);
	mat m2 = ones(sz, 2);
	mat m3 = randu(sz, 3);
	vec v1 = ones(sz);

	sp_mat sm1 = arma::sprandu(sz, 1, 0.3);
	sp_mat sm2 = arma::sprandu(sz, 2, 0.3);
	sp_mat sm3 = arma::sprandu(sz, 3, 0.3);

	mat K = join<mat>( { {v1, m2, m3}, {m3, m2, m1} } );
	K.print();

	sp_mat sK = join<sp_mat>( { {sm1, sm2, sm3}, {sm3, sm2, sm1} } );
	sK.print();

	std::cout << std::endl;


	/*
	mat M = joinr(m1, m2, m3);
	M.print();

	std::cout << std::endl;

	mat N = joinr2({m1, m2, m3});
	N.print();

	std::cout << std::endl;

	mat K = join2( { {v1, m2, m3}, {m3, m2, m1} } );
	K.print();

	std::cout << std::endl;

	mat L = join( {m1,m2,m3}, {m3,m2,m1} );
	L.print();
	*/

	return 0;
}
