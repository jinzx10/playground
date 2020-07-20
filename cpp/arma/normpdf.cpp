#include <iostream>
#include <armadillo>

using namespace arma;
using namespace std;

int main() {

	vec a(5, fill::randu);
	double sigma = 0.7;
	double mu = 0.22;

	arma::normpdf(a, mu, sigma).print();

	vec b = 1.0/datum::sqrt2pi/sigma*exp(-0.5*square((a-mu)/sigma));
	b.print();

	return 0;
}
