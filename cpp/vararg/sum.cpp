#include <cstdarg>
#include <iostream>
#include <armadillo>

using namespace arma;

int isum(int n_args, ...) {
	va_list args;
	va_start(args, n_args);
	int result = 0.0;
	for (int i = 0; i != n_args; ++i)
		result += va_arg(args, int);
	va_end(args);
	return result;
}

double dsum(int n_args, ...) {
	va_list args;
	va_start(args, n_args);
	double result = 0.0;
	for (int i = 0; i != n_args; ++i) {
		std::cout << result << "   ";
		result += va_arg(args, double);
		std::cout << result << std::endl;
	}
	va_end(args);
	return result;
}

template <typename ...Ts>
double sum(double const& var, Ts const& ...vars) {
	return var + sum(vars...);
}

double sum(double const& x) {
	return x;
}

int main() {
	/*
	std::cout << isum(2,1,2) << std::endl;
	std::cout << isum(3,1,2,3) << std::endl;
	std::cout << isum(4,1,2,3,4) << std::endl;
	std::cout << dsum(2,1.,2.) << std::endl;
	std::cout << dsum(3,1.,2.,3.) << std::endl;
	std::cout << dsum(4,1.,2.,3.,4.) << std::endl;
	*/
	std::cout << dsum(2,1,2.) << std::endl;
	std::cout << sum(1, 2.2, 3.3, 4) << std::endl;

	return 0;
}
