#ifndef __fdcoef_h
#define __fdcoef_h

template <unsigned int N>
double power(double const& x) {
	return x * power<N-1>(x);
}

template <>
double power<0>(double const&) {
	return 1;
}

template <unsigned int N>
unsigned int factorial() {
	return N * factorial<N-1>();
}

template <>
unsigned int factorial<0>() {
	return 1;
}

#include <armadillo>

inline arma::vec fdcoef(unsigned int od, unsigned int og) {
	arma::uvec P = arma::regspace<arma::uvec>(-og, 1, og);
	arma::uword M = P.n_elem;
	arma::vec z = arma::zeros(M);
	z(od) = 1;
	arma::mat D = ;
	return arma::solve(D, z);

}

#endif
