#include <iostream>
#include <functional>
#include <cmath>

double sumsqr(double n1, double n2, double n3) {
	return std::pow(n1,2) + std::pow(n2,2) + std::pow(n3,2);
}

int main() {

	double c = 0.2;
	auto f = std::bind(sumsqr, std::placeholders::_1, c, 0);
	std::function<double(double)> g = std::bind(sumsqr, std::placeholders::_1, std::cref(c), 0);

	std::cout << f(1) << std::endl;
	std::cout << g(1) << std::endl;

	c = 0.3;

	std::cout << f(1) << std::endl;
	std::cout << g(1) << std::endl;

	std::cout << typeid(f).name() << std::endl;
	std::cout << typeid(g).name() << std::endl;

	return 0;
}
