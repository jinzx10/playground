#include <iostream>
#include <armadillo>
#include <chrono>

using iclock = std::chrono::high_resolution_clock;
using namespace arma;

mat repadd(vec const& x, rowvec const& y) {
	return repmat(x,1,y.n_elem) + repmat(y,x.n_elem,1);
}

mat muladd(vec const& x, rowvec const& y) {
	return x*ones(1,y.n_elem) + ones(x.n_elem)*y;
}

mat foradd(vec const& x, rowvec const& y) {
	mat z(x.n_elem, y.n_elem);
	for (uword j = 0; j != y.n_elem; ++j)
		z.col(j) = x + y(j);
	return z;
}


int main() {
	uword sz1 = 800;
	uword sz2 = 159600;
	uword n_trials = 10;

	arma::vec a = randu(sz1);
	arma::rowvec b = randu<rowvec>(sz2);

	iclock::time_point start;
	std::chrono::duration<double> dur;

	start = iclock::now();
	for (uword i = 0; i != n_trials; ++i) {
		mat c = repadd(a,b);
	}
	dur = iclock::now() - start;
	std::cout << "repadd time elapsed = " << dur.count() << std::endl;

	start = iclock::now();
	for (uword i = 0; i != n_trials; ++i) {
		mat c = muladd(a,b);
	}
	dur = iclock::now() - start;
	std::cout << "muladd time elapsed = " << dur.count() << std::endl;

	start = iclock::now();
	for (uword i = 0; i != n_trials; ++i) {
		mat c = foradd(a,b);
	}
	dur = iclock::now() - start;
	std::cout << "foradd time elapsed = " << dur.count() << std::endl;

	start = iclock::now();
	for (uword i = 0; i != n_trials; ++i) {
		mat z(a.n_elem, b.n_elem);
		for (uword j = 0; j != sz2; ++j) {
			z.col(j) = a + b(j);
		}
	}
	dur = iclock::now() - start;
	std::cout << "for time elapsed = " << dur.count() << std::endl;

	return 0;
}
