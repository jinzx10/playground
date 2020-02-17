#include <iostream>
#include <armadillo>
#include "../utility/stopwatch.h"
#include "../utility/readargs.h"


struct Tq
{
	// parametrized constructor
	Tq(arma::mat const& a) : A(a) { std::cout << "parametrized copy constructor called" << std::endl; }
	Tq(arma::mat&& a) : A(std::move(a)) { std::cout << "parametrized move constructor called" << std::endl; }

	// copy constructor
	Tq(Tq const& tq) : A(tq.A) { std::cout << "copy constructor called" << std::endl; }

	// move constructor
	Tq(Tq&& tq) : A(std::move(tq.A)) { std::cout << "move constructor called" << std::endl; }

	arma::mat A;

	void clear() { A.set_size(0,0); }
};


int main(int, char** argv) {

	Stopwatch sw;
	arma::uword sz = 0;
	readargs(argv, sz);

	arma::mat a = arma::zeros(sz,sz);

	sw.run();
	Tq tq(arma::zeros(sz,sz));
	sw.report();
	std::cout << std::endl;
	//tq.clear();
	sw.reset();

	sw.run();
	Tq tq2(a);
	sw.report();
	std::cout << std::endl;
	tq2.clear();
	sw.reset();

	sw.run();
	Tq tq3(tq);
	sw.report();
	std::cout << std::endl;
	tq3.clear();
	tq.clear();
	sw.reset();

	sw.run();
	Tq tq4(Tq(arma::zeros(sz,sz)));
	sw.report();
	std::cout << std::endl;
	tq4.clear();
	sw.reset();


	return 0;
}
