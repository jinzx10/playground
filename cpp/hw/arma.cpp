#include <iostream>
#include <armadillo>
#include <chrono>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;

struct Test
{
	Test(uword n) : a(randu(n,n)) {b = mat(a.memptr(), a.n_rows, a.n_cols, false);}

	mat a;
	arma::subview<double> a01() {
		return a.cols(0,1);
	}

	mat b;

};

void f(mat& a) {
	a.print();
}

int main() {
	mat a = eye(10,10);

	mat b = a * 3;
	b.print();

	return 0;
}
