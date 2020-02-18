#include <iostream>
#include <armadillo>
#include "../utility/grad.h"
#include <functional>

using namespace arma;
using namespace std::placeholders;

int sum(int x, int y, int z) {
	return x+10*y+100*z;
}

double tq(double x) {
	return std::abs(x);
}

int main() {
	arma::vec v = {1.5, 3.7};

	auto dtq = grad(tq);
	std::cout << dtq(1) << std::endl
		<< dtq(-1) << std::endl
		<< dtq(0) << std::endl;

	std::function<double(arma::vec)> f = [] (arma::vec const& v) {
		return v(0)*v(0)+v(1)*v(1)*2 + v(0)*v(1) + exp(v(0));
	};

	std::function<double(std::vector<double>)> g = [] (std::vector<double> const& v) {return v[0]*v[0]*0.5+v[1]*v[1]*4.0; };

	auto grad_f = grad(f,1);
	grad_f(v).print();

	auto grad_g = grad(g);
	auto val = grad_g(std::vector<double>{1.1,2.2});

	for (auto& e : val)
		std::cout << e << std::endl;

	
	auto s2 = std::bind(::sum, _2, 7, _1);

	std::cout << s2(1,5) << std::endl;

	return 0;
}
