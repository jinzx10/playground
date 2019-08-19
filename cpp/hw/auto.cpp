#include <functional>
#include <iostream>
#include <type_traits>
#include <ccomplex>

std::complex<double> I(0,1);

template <typename T>
using Func = std::function<double(T)>;

template <typename T>
auto merge(Func<T> f1, Func<T> f2) {
	return [f1,f2] (double x) -> std::complex<double> {
		return f1(x) + I*f2(x);
	};
}

double sqr(double x) {
	return x*x;
}

int main() {

	Func<double> f1 = [](double x) {return x*x;};
	Func<double> f2 = [](double x) {return 0.5*x;};

	auto g = merge(f1,f2);

	auto s = sqr;

	std::cout << g(0.5) << std::endl;
	std::cout << typeid(g).name() << std::endl;
	std::cout << typeid(s).name() << std::endl;

	return 0;
}
