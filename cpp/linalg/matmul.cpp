#include <armadillo>
#include "../utility/widgets.h"

using namespace arma;

int main(int, char** argv) {

	uword sz = 0;
	uword nt = 0;
	readargs(argv, sz, nt);

	auto f = [] (mat const& a, mat const& b) -> mat {return a*b;};

	mat a = randn(sz, sz);
	mat b = randn(sz, sz);

	Stopwatch sw;

	sw.timeit("arma matrix multiplication", nt, f, a, b);

	return 0;

}
