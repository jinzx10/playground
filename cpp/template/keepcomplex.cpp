#include <iostream>
#include <type_traits>
#include <complex>
#include <armadillo>

template <bool is_cplx> using num_t = typename std::conditional<is_cplx, std::complex<double>, double>::type;

// function template
template <bool is_cplx> num_t<is_cplx> keep_cplx(std::complex<double> const& z) { return z; }
template <> num_t<false> keep_cplx<false>(std::complex<double> const& z) { return z.real(); }

template <bool is_cplx, typename T> struct KeepCplx { static auto value(T const& z) {return z;} };
template <typename T> struct KeepCplx<false,T> { static auto value(T const& z) {return real(z);} }; // ADL, std:: or arma::


int main() {

	std::complex<double> z(1.1, 2.2);
	arma::Col<std::complex<double>> zv = {z,3.0*z};
	arma::cx_vec2 fzv = zv;

	std::cout << keep_cplx<false>(z) << std::endl;
	std::cout << keep_cplx<true>(z) << std::endl;

	std::cout << KeepCplx<false, num_t<true>>::value(z) << std::endl;
	std::cout << KeepCplx<true, num_t<true>>::value(z) << std::endl;
	std::cout << KeepCplx<false, arma::cx_vec>::value(zv) << std::endl;
	std::cout << KeepCplx<true, arma::cx_vec>::value(zv) << std::endl;
	std::cout << KeepCplx<false, arma::cx_vec2>::value(fzv) << std::endl;
	std::cout << KeepCplx<true, arma::cx_vec2>::value(fzv) << std::endl;



	return 0;
}
