#include <armadillo>
#include <chrono>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {

	uword sz = 0;
	uword nt = 0;

	if (argc < 3) {
		std::cerr << "please provide two arguments: size and number of trials." << std::endl;
		return -1;
	}

	std::stringstream ss;
	ss << argv[1] << ' ' << argv[2];
	ss >> sz >> nt;
	
	mat a = randn(sz, sz);
	a += a.t();

	mat b(sz, sz);

	std::cout << "sz = " << sz << std::endl;
	std::cout << "nt = " << nt << std::endl;

	iclock::time_point start = iclock::now();
	std::chrono::duration<double> dur;

	for (uword i = 0; i != nt; ++i) {
		b = a*a;
	}

	dur = iclock::now() - start;

	std::cout << "matrix size = " << sz 
		<< "      average time elapsed = " << dur.count() / nt 
		<< " seconds for " << nt << " trials." << std::endl;

	return 0;

}
