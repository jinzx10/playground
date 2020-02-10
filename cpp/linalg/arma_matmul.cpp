#include <armadillo>
#include <chrono>

using namespace arma;
using iclock = std::chrono::high_resolution_clock;

int main() {

	uword sz = 4000;
	mat a = randn(sz, sz);

	vec val;
	mat vec;

	iclock::time_point start = iclock::now();
	std::chrono::duration<double> dur;

	mat b = a * a;

	dur = iclock::now() - start;

	std::cout << "matrix size = " << sz << "      time elapsed = " << dur.count() << std::endl;

	return 0;

}

