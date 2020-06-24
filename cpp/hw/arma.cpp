#include <iostream>
#include <armadillo>
#include <chrono>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;


int main() {

	rowvec a = randu(1,3);
	vec b = randu(3,1);

	cout << dot(a,b) << endl;

	cout << as_scalar(a*b) << endl;

	return 0;
}
