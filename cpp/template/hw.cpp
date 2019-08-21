#include <type_traits>
#include <iostream>
#include <complex>
#include <armadillo>

template <bool is_cplx>
using num_t = typename std::conditional<is_cplx, std::complex<double>, double>::type;

template <bool is_cplx>
num_t<is_cplx> keep_cplx(std::complex<double>& z) { return z; }

template <>
num_t<false> keep_cplx<false>(std::complex<double>& z) { return z.real(); }


int main() 
{
	arma::Col<double>::fixed<0> v0;
	std::cout << typeid(v0).name() << std::endl;
	return 0;
}
	
