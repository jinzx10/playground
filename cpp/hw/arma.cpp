#include <iostream>
#include <armadillo>
#include <chrono>

using iclock = std::chrono::high_resolution_clock;
using namespace arma;

int main() {
	uword sz1 = 4000;
	uword sz2 = 4000;
	uword n_trials = 10;

	mat a = randn<mat>(sz1, sz2);
	mat b = zeros(sz1, sz2);

	iclock::time_point start;
	std::chrono::duration<double> dur;

	start = iclock::now();
	for (uword i = 0; i != n_trials; ++i) {
		b = exp(a);
	}
	dur = iclock::now() - start;
	std::cout << "exp(a) elapsed = " << dur.count() << std::endl;

	return 0;
}
