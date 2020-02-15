#include <iostream>
#include <complex>
#include <armadillo>
#include <string>
#include "../utility/stopwatch.h"


int main() {

	Stopwatch sw;

	int sz = 1000;
	arma::mat a = arma::ones(sz,sz);
	sw.run();

	auto b = exp(a);
	sw.report();

	arma::mat c = b;
	sw.report();


	return 0;
}
