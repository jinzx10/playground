#include <iostream>
#include <armadillo>
#include <chrono>

using iclock = std::chrono::high_resolution_clock;

double min(double const& i) {
	return i;
}

template <typename ...Ts>
double min(double const& i, Ts const& ...args) {
	double tmp = min(args...);
	return ( i < tmp ) ? i : tmp;
}

int main() {
	double a = 3.0;
	int n_times = 1000000;
	iclock::time_point start = iclock::now();
	for (int i = 0; i != n_times; ++i) {
		double b = arma::min(arma::vec{1,2,-a/2,-1});
	}
	std::chrono::duration<double> dur = iclock::now() - start;
	std::cout << "arma min time elapsed = " << dur.count() << std::endl;

	start = iclock::now();
	for (int i = 0; i != n_times; ++i) {
		double c = min(1,3,-a/2,-1);
	}
	dur = iclock::now() - start;
	std::cout << "min time elapsed = " << dur.count() << std::endl;

	return 0;
}
