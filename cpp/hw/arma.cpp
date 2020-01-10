#include <iostream>
#include <armadillo>
#include <type_traits>

using namespace arma;


int main() {
	arma::mat a = randu(3,3);

	a.print();

	return 0;
}
