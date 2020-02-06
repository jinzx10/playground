#include <iostream>
#include <armadillo>
#include <chrono>

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;
using iclock = std::chrono::high_resolution_clock;


int main() {

	vec v = {1,2,3.14};
	std::cout << v.back() << std::endl;

	v.resize(v.n_elem+1);
	v.back() = 6.28;
	v.print();

	std::cout << GET_VARIABLE_NAME(v) << std::endl;

	
	v.save("v.dat", arma::raw_binary);

	vec u;
	u.load("v.dat");
	u.print();
	


	return 0;
}
