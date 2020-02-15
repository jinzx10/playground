#include "stopwatch.h"
#include <cstdlib>
#include <string>
#include <armadillo>

int main() {

	Stopwatch sw;

	int sz = 1000;
	arma::mat a = arma::ones(sz,sz);
	auto f = [] (arma::mat const& a) -> arma::mat { return arma::exp(a); };

	sw.timeit(f, a);

	sw.run();
	arma::mat b = arma::exp(a);
	sw.report();


	return 0;
}
