#include <iostream>
#include <armadillo>
#include "../utility/stopwatch.h"

template <typename F, typename ...Args>
void show(F f, Args ...args) {
	std::cout << "functor" << std::endl;
	std::cout << f(args...) << std::endl;
}

/*
template <typename R, typename ...Args>
void show(R(*F)(Args...), Args ...args) {
	std::cout << "function pointer" << std::endl;
	std::cout << F(args...) << std::endl;
}
*/

int add(int x, int y) {
	return x+y;
}

double add(double x, double y) {
	return x+y;
}

template <typename ...Args>
struct get
{
	auto static wrapper = [](Args& ...args) {};

	template <typename R>
	static const R func( R(*)(Args...) ); 
	
};

template <typename F, typename ...Args>
using return_t = decltype( std::declval<F>()( std::declval<Args>()... ) );

int main() {

	std::cout << show(add, 3.0, 3.0) << std::endl;

	

	return 0;
}
