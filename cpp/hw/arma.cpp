#include <armadillo>
#include <type_traits>

template <typename T>
typename std::enable_if<arma::is_arma_type<T>::value, T>::type zero() {
	return T(arma::fill::zeros);
}

template <typename T>
typename std::enable_if<!arma::is_arma_type<T>::value, T>::type zero() {
	return 0.0;
}

int main() {
	auto cv = zero<arma::cx_vec4>();
	cv.print();

	auto z = zero<std::complex<double>>();
	std::cout << z << std::endl;
	return 0;
}
