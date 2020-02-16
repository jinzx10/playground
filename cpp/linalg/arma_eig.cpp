#include <armadillo>
#include <chrono>
#include <sstream>
#include "../utility/stopwatch.h"
#include "../utility/readargs.h"

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {

	uword sz = 0;

	readargs(argv, sz);

	auto f = [](const mat& a) -> arma::vec { return arma::eig_sym(a); };
	//auto g = [](const mat& a) { return arma::eig_sym(a); };

	Stopwatch sw;

	mat a = randn(sz,sz);
	a += a.t();

	sw.timeit("eigval", f, a);
	//sw.timeit(g, a);

	return 0;

}
