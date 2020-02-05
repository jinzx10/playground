#include <iostream>
#include <armadillo>
#include <chrono>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

template <uword N, typename eT>
void mass_zeros(arma::Mat<eT>& m) {
	m.zeros(N);
}

template <uword N, typename eT, typename ...Ts>
void mass_zeros(arma::Mat<eT>& m, Ts& ...args) {
	m.zeros(N);
	mass_zeros<N>(args...);
}

template <uword M, uword N, typename eT>
void mass_zeros(arma::Mat<eT>& m) {
	m.zeros(M, N);
}

template <uword M, uword N, typename eT, typename ...Ts>
void mass_zeros(arma::Mat<eT>& m, Ts& ...args) {
	m.zeros(M,N);
	mass_zeros<M,N>(args...);
}



int main() {
	mat a,b,c;
	vec d;
	mass_zeros<5>(a,d);
	mass_zeros<3,4>(b,c);

	a.print();
	std::cout << std::endl;
	b.print();
	std::cout << std::endl;
	c.print();
	std::cout << std::endl;
	d.print();
	std::cout << std::endl;

	return 0;
}
