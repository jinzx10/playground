#include <chrono>
#include <iostream>
#include <armadillo>

using iclock = std::chrono::high_resolution_clock;

struct Tq
{
	Tq(arma::mat const& a) : A(a) {}
	Tq(arma::mat&& a) { A = std::move(a); }
	arma::mat A;
};

struct Tq_var
{
	Tq_var(arma::mat const& a) : A(a) {}
	Tq_var(arma::mat&& a) : A(a) { }
	arma::mat A;
};


struct Tq_old
{
	Tq_old(arma::mat const& a) : A(a) {}
	arma::mat A;
};

int main() {
	size_t sz = 5000;

	iclock::time_point start;
	std::chrono::duration<double> dur;

	start = iclock::now();
	Tq tq(arma::zeros(sz,sz));
	dur = iclock::now() - start;
	std::cout << "move: " << dur.count() << std::endl;

	start = iclock::now();
	Tq_old tq_old(arma::zeros(sz,sz));
	dur = iclock::now() - start;
	std::cout << "copy: " << dur.count() << std::endl;

	return 0;
}
