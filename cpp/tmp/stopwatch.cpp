#include <cstdlib>
#include <string>
#include <armadillo>
#include "../utility/stopwatch.h"
#include "../utility/readargs.h"

int main(int, char**argv) {

	Stopwatch sw;

	int sz, nt;

	readargs(argv, sz, nt);

	std::cout << "sz = " << sz << std::endl;
	std::cout << "nt = " << nt << std::endl;

	arma::mat a = arma::ones(sz,sz);
	auto f = [] (arma::mat const& a) -> arma::mat { return arma::exp(a); };
	auto matmul = [] (arma::mat const& m, arma::mat const& n) -> arma::mat {return m*n;};

	auto h = [&a] () -> arma::vec { return arma::eig_sym(a); };

	sw.timeit(f, a);
	sw.timeit(matmul, a, a);

	sw.timeit("eig", f, a);
	sw.timeit("matmul", matmul, a, a);

	sw.timeit(nt, f, a);
	sw.timeit(nt, matmul, a, a);

	sw.timeit("eig", nt, f, a);
	sw.timeit("matmul", nt, matmul, a, a);

	sw.timeit("test eig", h);

	sw.run();
	arma::mat b = arma::exp(a);
	sw.report("single exp");

	sw.reset();
	sw.run();
	arma::mat c = a*b;
	sw.report("single matmul");


	return 0;
}
