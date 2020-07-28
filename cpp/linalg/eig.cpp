#include <armadillo>
#include <sstream>
#include <chrono>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main(int, char** argv) {

	uword sz = 0;
	uword nt = 0;

	std::stringstream ss;
	ss << argv[1];
	ss >> sz;
	ss.clear();
	ss.str("");

	ss << argv[2];
	ss >> nt;
	ss.clear();
	ss.str("");

	std::cout << sz << std::endl
		<< nt << std::endl;

	mat a = randu(sz, sz);
	a += a.t();

	mat evec(sz,sz);
	vec eval(sz);

	iclock::time_point start = iclock::now();
	for (int i = 0; i != nt; ++i) {
		eig_sym(eval, evec, a);
	}
	std::chrono::duration<double> dur = iclock::now() - start;
	std::cout << "average time elapsed: " << dur.count()/nt << " seconds" << std::endl;

	return 0;

}
