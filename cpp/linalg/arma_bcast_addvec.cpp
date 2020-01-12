#define NAME(var) (#var)

#include <iostream>
#include <armadillo>
#include <chrono>

using iclock = std::chrono::high_resolution_clock;
using namespace arma;

struct timer
{
	static void call(std::function<mat(vec,rowvec)> f, vec const& a, rowvec const& b, uword const& n = 10) {
		iclock::time_point start = iclock::now();
		for (uword i = 0; i != n; ++i)
			mat c = f(a,b);
		std::chrono::duration<double> dur = iclock::now() - start;
		std::cout << n << " times, time elapsed = " << dur.count() << std::endl;
	}
};

mat repadd(vec const& x, rowvec const& y) {
	return repmat(x,1,y.n_elem) + repmat(y,x.n_elem,1);
}

mat muladd(vec const& x, rowvec const& y) {
	return x*ones(1,y.n_elem) + ones(x.n_elem)*y;
}

mat coladd(vec const& x, rowvec const& y) {
	mat z(x.n_elem, y.n_elem);
	for (uword j = 0; j != y.n_elem; ++j)
		z.col(j) = x + y(j);
	return z;
}

mat rowadd(vec const& x, rowvec const& y) {
	mat z(x.n_elem, y.n_elem);
	for (uword i = 0; i != x.n_elem; ++i)
		z.row(i) = x(i) + y;
	return z;
}

mat foradd(vec const& x, rowvec const& y) {
	mat z(x.n_elem, y.n_elem);
	for (uword i = 0; i != x.n_elem; ++i)
		for (uword j = 0; j != y.n_elem; ++j)
			z(i,j) = x(i) + y(j);
	return z;
}


int main() {
	uword sz1 = 10000;
	uword sz2 = 10000;
	uword n_trials = 10;

	arma::vec a = randu(sz1);
	arma::rowvec b = randu<rowvec>(sz2);

	std::cout << "repadd: "; 
	timer::call(repadd, a, b);

	std::cout << "muladd: "; 
	timer::call(muladd, a, b);

	std::cout << "coladd: "; 
	timer::call(coladd, a, b);

	std::cout << "rowadd: "; 
	timer::call(rowadd, a, b);

	std::cout << "foradd: "; 
	timer::call(foradd, a, b);


	return 0;
}
