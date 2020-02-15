#include "stopwatch.h"
#include <cstdlib>
#include <string>
#include <armadillo>

int main() {

	Stopwatch sw;

	int sz = 200;
	arma::mat a = arma::ones(sz,sz);
	auto f = [] (arma::mat const& a) -> arma::mat { return arma::exp(a); };
	auto matmul = [] (arma::mat const& m, arma::mat const& n) -> arma::mat {return m*n;};

	sw.timeit<0>(f, a);
	sw.timeit<10>(f, a);
	sw.timeit<100>(f, a);

	sw.timeit<0>(matmul, a, a);
	sw.timeit<10>(matmul, a, a);
	sw.timeit<100>(matmul, a, a);

	sw.run();
	arma::mat b = arma::exp(a);
	sw.report("single exp");

	sw.reset();
	sw.run();
	arma::mat c = a*b;
	sw.report("single matmul");


	return 0;
}
