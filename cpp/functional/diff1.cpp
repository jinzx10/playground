#include <vector>
#include <functional>
#include <iostream>

double const h = 1e-6;

using Func = std::function<double(double)>;

double para1(double x) { return x*x;}
double para2(double x) { return (x-1)*(x-1);}

Func diff1(Func const& f) {
	return [f](double x) {
		return (f(x+h) - f(x-h))/2.0/h;
	};
}

int main() {

	std::cout << diff1(para1)(0.5) << std::endl;
	std::cout << diff1(para2)(0.5) << std::endl;

	std::vector<Func> vf = {diff1(para1), diff1(para2)};

	double sum = 0;

	for (auto& c : vf) 
		sum += c(0.5);


	return 0;
}

