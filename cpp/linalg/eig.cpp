#include <armadillo>
#include "../utility/widgets.h"

using namespace arma;

int main(int, char** argv) {

	uword sz = 0;
	uword nt = 0;
	Stopwatch sw;

	readargs(argv, sz, nt);
	std::cout << sz << std::endl
		<< nt << std::endl;

	auto f = [](const mat& a) -> arma::vec { return arma::eig_sym(a); };

	mat a = randn(sz,sz);
	a += a.t();

	sw.timeit("arma::eig_sym", nt, f, a);

	return 0;

}
