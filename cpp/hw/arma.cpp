#include <iostream>
#include <armadillo>
#include <type_traits>

using namespace arma;


int main() {
	arma::mat a = arma::randu(5,4);
	a.print();

	arma::mat q,r;

	arma::qr(q,r,a);

	q.print();
	r.print();
	(q*r).print();

	return 0;
}
