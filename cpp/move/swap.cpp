#include <iostream>
#include <armadillo>
#include <chrono>

using iclock = std::chrono::high_resolution_clock;

void copy_swap(arma::mat& a, arma::mat& b) {
	arma::mat tmp(a);
	a = b;
	b = tmp;
}

void move_swap(arma::mat& a, arma::mat& b) {
	std::cout << "a memptr before move: " << a.memptr() << std::endl;
	std::cout << "b memptr before move: " << b.memptr() << std::endl;
	arma::mat tmp = std::move(a);
	std::cout << "a memptr after move: " << a.memptr() << std::endl;
	std::cout << "tmp memptr: " << tmp.memptr() << std::endl;
	a = std::move(b);
	std::cout << "b memptr after move: " << b.memptr() << std::endl;
	std::cout << "a memptr after assign: " << a.memptr() << std::endl;
	b = std::move(tmp);
	std::cout << "tmp memptr after move: " << tmp.memptr() << std::endl;
	std::cout << "b memptr after assign: " << b.memptr() << std::endl;
}

int main() {

	size_t sz = 4000;

	arma::mat a = arma::randu(sz,sz);
	arma::mat b = arma::randu(sz,sz);

	iclock::time_point start;
	std::chrono::duration<double> dur;

	start = iclock::now();
	move_swap(a,b);
	dur = iclock::now() - start;
	std::cout << std::endl << "move-swap time elapsed = " << dur.count() << std::endl;

	start = iclock::now();
	copy_swap(a,b);
	dur = iclock::now() - start;
	std::cout << "copy-swap time elapsed = " << dur.count() << std::endl << std::endl;

	start = iclock::now();
	copy_swap(a,b);
	dur = iclock::now() - start;
	std::cout << "copy-swap time elapsed = " << dur.count() << std::endl << std::endl;

	start = iclock::now();
	move_swap(a,b);
	dur = iclock::now() - start;
	std::cout << std::endl << "move-swap time elapsed = " << dur.count() << std::endl;



	return 0;
}
