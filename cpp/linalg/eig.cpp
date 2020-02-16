#include <armadillo>
#include "../utility/stopwatch.h"
#include "../utility/readargs.h"

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main(int, char** argv) {

	uword sz = 0;
	uword nt = 0;
	Stopwatch sw;

	readargs(argv, sz, nt);

	auto f = [](const mat& a) -> arma::vec { return arma::eig_sym(a); };

	mat a = randn(sz,sz);
	a += a.t();

	sw.timeit("arma::eig_sym", nt, f, a);

	return 0;

}
