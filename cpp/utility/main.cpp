#include <iostream>
#include <armadillo>
#include "grad.h"

using namespace arma;

int main() {
	arma::vec v = {1.5, 3.7};

	auto func = [](arma::vec const& v) {return v(0)*v(0)+v(1)*v(1)*2; };


	auto grad_f = grad(func);

	grad_f({2.0, 0.5}).print();

	uword i = 0;
	auto fi = [func, i] (arma::vec const& v) {return func();};


	return 0;
}
