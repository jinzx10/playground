#include <iostream>
#include <armadillo>
#include "grad.h"
#include <functional>

using namespace arma;
using namespace std::placeholders;

int sum(int x, int y, int z) {
	return x+10*y+100*z;
}

int main() {
	arma::vec v = {1.5, 3.7};

	std::function<double(arma::vec)> f = [] (arma::vec const& v) {return v(0)*v(0)+v(1)*v(1)*2; };
	std::function<double(std::vector<double>)> g = [] (std::vector<double> const& v) {return v[0]*v[0]*0.5+v[1]*v[1]*4.0; };

	auto grad_f = grad(f);
	grad_f(v).print();

	auto grad_g = grad(g);
	auto val = grad_g(std::vector<double>{1.1,2.2});

	for (auto& e : val)
		std::cout << e << std::endl;

	
	auto s2 = std::bind(::sum, _2, 7, _1);

	std::cout << s2(1,5) << std::endl;

	return 0;
}
