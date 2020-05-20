#include <iostream>
#include <armadillo>
#include <chrono>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;


int main() {

	vec a(10, fill::zeros);
	a(4) = datum::nan;
	a(7) = datum::inf;
	a(9) = -1/0.0;
	a(2) = 1/0.0;

	a.print();

	std::cout << std::endl;

	a.replace(datum::inf, 1);
	a.print();
	std::cout << std::endl;

	a(arma::find_nonfinite(a)).ones();
	a.print();

	return 0;
}
