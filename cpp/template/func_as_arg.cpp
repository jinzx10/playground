#include <iostream>
#include <armadillo>
#include "../utility/stopwatch.h"

template <typename F, typename ...Args>
void show(F f, Args ...args) {
	std::cout << "functor" << std::endl;
	std::cout << f(args...) << std::endl;
}

template <typename R, typename ...Args>
void show(R(*F)(Args...), Args ...args) {
	std::cout << "function pointer" << std::endl;
	std::cout << F(args...) << std::endl;
}

int add(int x, int y) {
	return x+y;
}

template <typename ...Args>
struct get
{
	template <typename R>
	static const R func( R(*)(Args...) ); 
};

template <typename F, typename ...Args>
using return_t = decltype( std::declval<F>()( std::declval<Args>()... ) );

int main() {
	Stopwatch sw;

	auto minus = [](double x, double y) {return x-y;};
	show(add, 2, 3);
	show(minus, 8, 6);

	auto power = [](arma::mat const& a) {return arma::exp(a);};

	show<double,double,double>(std::pow, 3.0, 3.0);
	//show(std::pow, x, y); // cannot deduce template parameter R

	std::cout << typeid(add).name() << std::endl;
	
	auto f = [] (arma::mat a) {return arma::exp(a);};

	std::cout << typeid( return_t<decltype(f),arma::mat> ).name() << std::endl;


	return 0;
}
