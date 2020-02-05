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

template <typename T>
T joinr(T const& m1, T const& m2) {
	return join_rows(m1, m2);
}

template <typename T, typename ...Ts>
T joinr(T const& m, Ts const& ...ms) {
	return join_rows( m, joinr(ms...) );
}

template <int N>
int factorial() {
	return N*factorial<N-1>();
}

//template <>
//unsigned int factorial<1>() {
//	return 1;
//}

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


	mat M = joinr<mat>(m1, m2, m3);
	M.print();

	std::cout << std::endl;

	sp_mat sM = joinr<sp_mat>(sm1, sm2, sm3);
	sM.print();

	std::cout << std::endl;

	mat mM = joinr<mat>(v1, m2, m3);
	mM.print();

	std::cout << std::endl;

	return 0;
}
