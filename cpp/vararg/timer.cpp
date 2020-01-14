#include <iostream>
#include <chrono>
#include <armadillo>

using iclock = std::chrono::high_resolution_clock;
using namespace arma;

template <typename F, typename ...Ts>
void timer(F f, Ts ...args) {
	auto start = iclock::now();
	f(args...);
	std::chrono::duration<double> dur = iclock::now() - start;
	std::cout << "time elapsed = " << dur.count() << std::endl;
}

int main() {
	auto f = [] (double x) {return 2*x;};
	auto matmul = [] (mat const& a, mat const& b) {return a*b;};

	timer(f, 3);

	int sz = 1000;
	timer(matmul, randu(sz,sz), randu(sz,sz));

	return 0;
}
