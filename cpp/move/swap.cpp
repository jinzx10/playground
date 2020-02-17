#include <iostream>
#include <armadillo>
#include <chrono>
#include "../utility/readargs.h"
#include "../utility/stopwatch.h"

void copy_swap(arma::mat& a, arma::mat& b) {
	arma::mat tmp(a);
	a = b;
	b = tmp;
}

void move_swap(arma::mat& a, arma::mat& b, bool info = false) {
	if (info) {
		std::cout << "a memptr before move: " << a.memptr() << std::endl;
		std::cout << "b memptr before move: " << b.memptr() << std::endl;
		std::cout << "move a to tmp" << std::endl;
	}
	arma::mat tmp = std::move(a);

	if (info) {
		std::cout << "a memptr after move: " << a.memptr() << std::endl;
		std::cout << "tmp memptr: " << tmp.memptr() << std::endl;
		std::cout << "move b to a" << std::endl;
	}
	a = std::move(b);

	if (info) {
		std::cout << "b memptr after move: " << b.memptr() << std::endl;
		std::cout << "a memptr after assign: " << a.memptr() << std::endl;
		std::cout << "move tmp to b" << std::endl;
	}
	b = std::move(tmp);

	if (info) {
		std::cout << "tmp memptr after move: " << tmp.memptr() << std::endl;
		std::cout << "b memptr after assign: " << b.memptr() << std::endl;
	}
}

int main(int, char**argv) {

	arma::uword sz = 0;
	readargs(argv, sz);

	Stopwatch sw;

	arma::mat a = arma::zeros(sz,sz);
	arma::mat b = arma::ones(sz,sz);


	sw.run();
	move_swap(a,b);
	sw.report("move swap");

	sw.reset();
	sw.run();
	copy_swap(a,b);
	sw.report("copy swap");

	sw.reset();
	sw.run();
	std::swap(a,b);
	sw.report("std::swap");

	sw.reset();
	sw.run();
	move_swap(a,b,true);
	sw.report("move swap (with info)");


	return 0;
}
