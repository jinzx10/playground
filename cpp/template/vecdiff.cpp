#include "aux.h"
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <array>
#include <complex>

double const pi = std::acos(-1);
double const dx = 1e-6;
std::complex<double> const I(0,1);


template <typename T, typename = void>
struct is_good_type : std::false_type {};

template <typename T>
struct is_good_type<T, std::enable_if_t<is_iterable<T>::value && std::is_copy_constructible<T>::value, void>> : std::true_type {};


template <bool is_cplx>
using is_cplx_t = typename std::conditional<is_cplx, std::complex<double>, double>::type;


template <typename T>
std::function<double(T)> func_real(std::function<std::complex<double>(T)> const& f) {
	return [f](T x) {
		return f(x).real();
	};
}

template <typename T>
std::function<double(T)> func_imag(std::function<std::complex<double>(T)> const& f) {
	return [f](T x) {
		return f(x).imag();
	};
}


template <typename T>
std::enable_if_t<is_good_type<T>::value, T> pt(T x, size_t const& dim, double const& dx) {
	auto it = std::begin(x);
	std::advance(it, dim);
	(*it) += dx;
	return x;
}


template <typename T, bool is_cplx = false>
std::enable_if_t<is_good_type<T>::value, std::function<is_cplx_t<is_cplx>(T, size_t)>> vecdiff(std::function<is_cplx_t<is_cplx>(T)> const& f) {
	return [f](T const& x, size_t const& dim) -> is_cplx_t<is_cplx> {
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
	auto h = [](std::vector<double> const& x) { return x[1]*std::exp(I*x[0]/3.0); };

	auto df = vecdiff<std::vector<double>>(f);
	auto dg = vecdiff<std::array<double,2>>(g);
	auto dh = vecdiff<std::vector<double>,true>(h);

	auto vdf = VecDiff<std::vector<double>>(f);
	auto vdg = VecDiff<std::array<double,2>>(g);

	std::cout << df({1,3}, 0) << std::endl;
	std::cout << dg({2,2}, 1) << std::endl;
	std::cout << dh({pi, 3}, 0) << std::endl;

	for (auto& c : vdf({1,3}))
		std::cout << c << std::endl;

	for (auto& c : vdg({2,2}))
		std::cout << c << std::endl;


	return 0;
}
