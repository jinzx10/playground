#include <cstdlib>
#include <string>
#include <armadillo>
#include "stopwatch.h"
#include "readargs.h"

int main(int, char**argv) {

	Stopwatch sw;

	int sz, nt;

	readargs(argv, sz, nt);

	arma::mat a = arma::ones(sz,sz);
	auto f = [] (arma::mat const& a) -> arma::mat { return arma::exp(a); };
	auto matmul = [] (arma::mat const& m, arma::mat const& n) -> arma::mat {return m*n;};

	/*
	sw.timeit<0>(f, a);
	sw.timeit<10>(f, a);
	sw.timeit<100>(f, a);

	sw.timeit<0>(matmul, a, a);
	sw.timeit<10>(matmul, a, a);
	sw.timeit<100>(matmul, a, a);
	*/


	/*
	sw.timeit(0, f, a);
	sw.timeit(1, f, a);
	sw.timeit(f, a);
	sw.timeit(100, f, a);

	sw.timeit(0, matmul, a, a);
	sw.timeit(1, matmul, a, a);
	sw.timeit(matmul, a, a);
	sw.timeit(100, matmul, a, a);
	*/

	sw.timeit(f, a);
	sw.timeit(matmul, a, a);

	sw.timeit("eig", f, a);
	sw.timeit("matmul", matmul, a, a);

	sw.timeit(nt, f, a);
	sw.timeit(nt, matmul, a, a);

	sw.timeit("eig", nt, f, a);
	sw.timeit("matmul", nt, matmul, a, a);

	sw.run();
	arma::mat b = arma::exp(a);
	sw.report("single exp");

	sw.reset();
	sw.run();
	arma::mat c = a*b;
	sw.report("single matmul");


	return 0;
}
