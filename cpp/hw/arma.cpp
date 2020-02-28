#include <iostream>
#include <armadillo>
#include <chrono>
#include "../utility/arma_helper.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;

int main() {

	mat z = zeros(1,1);
	mat o = ones(1,1);

	mat a = join({{z, mat{3}, o}});

	a.print();

	return 0;
}
