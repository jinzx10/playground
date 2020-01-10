#include <iostream>
#include <armadillo>

size_t numel(double const& ) {return 1;}

int main() {
	arma::mat a = arma::randu(3,3);
	double b = 1;

	std::cout << numel(a) << std::endl
		<< numel(b) << std::endl;

	a(arma::span(0,1),0).print();

	return 0;
}
