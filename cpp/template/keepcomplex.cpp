#include <iostream>
#include <type_traits>
#include <complex>

template <bool is_cplx = true>
struct KeepCplx { static std::complex<double> value(std::complex<double> z) {return z;} };

template <>
struct KeepCplx<false> { static double value(std::complex<double> z) {return z.real();} };

int main() {

	std::complex<double> z(1.1, 2.2);

	std::cout << KeepCplx<false>::value(z) << std::endl;
	std::cout << KeepCplx<true>::value(z) << std::endl;

	return 0;
}
