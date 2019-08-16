#include "aux.h"
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <array>

double const dx = 1e-6;

template <typename T, typename = void>
struct is_good_type : std::false_type {};

template <typename T>
struct is_good_type<T, std::enable_if_t<is_iterable<T>::value && std::is_copy_constructible<T>::value, void>> : std::true_type {};

template <typename T>
std::enable_if_t<is_good_type<T>::value, T> pt(T x, size_t const& dim, double const& dx) {
	auto it = std::begin(x);
	std::advance(it, dim);
	(*it) += dx;
	return x;
}

template <typename T>
std::enable_if_t<is_good_type<T>::value, std::function<double(T, size_t)>> vecdiff(std::function<double(T)> const& f) {
	return [f](T const& x, size_t const& dim) {
		return ( f(pt(x, dim, dx)) - f(pt(x, dim, -dx)) ) / dx / 2.0;
	};
}

template <typename T>
std::enable_if_t<is_good_type<T>::value, std::function<T(T)>> VecDiff(std::function<double(T)> const& f) {
	return [f](T const& x) {
		T dfx = x;
		auto df = vecdiff(f);
		for (auto& val : dfx) {
			auto dim = &val - &*std::begin(dfx);
			val = df(x, dim);
		}
		return dfx;
	};
}

int main() {

	auto f = [](std::vector<double> const& x) { return std::pow(x[0],2) + std::pow(x[1],3) + x[0]*x[1]; };
	auto g = [](std::array<double, 2> const& x) { return x[0]*x[1] + std::pow(x[1],-1); };

	auto df = vecdiff<std::vector<double>>(f);
	auto dg = vecdiff<std::array<double,2>>(g);

	auto vdf = VecDiff<std::vector<double>>(f);
	auto vdg = VecDiff<std::array<double,2>>(g);

	std::cout << df({1,3}, 0) << std::endl;
	std::cout << dg({2,2}, 1) << std::endl;

	for (auto& c : vdf({1,3}))
		std::cout << c << std::endl;

	for (auto& c : vdg({2,2}))
		std::cout << c << std::endl;


	return 0;
}
