#include <armadillo>
#include <type_traits>

using namespace arma;

template <typename T>
typename std::enable_if<arma::is_arma_type<T>::value, T>::type zero() {
	return T(arma::fill::zeros);
}

template <typename T>
typename std::enable_if<!arma::is_arma_type<T>::value, T>::type zero() {
	return 0.0;
}


int main() {
	size_t nx = 10;
	vec x = linspace<vec>(0, 1, nx);
	cx_vec z = linspace<cx_vec>(0, 1, nx);

	cx_mat dc = zeros<cx_mat>(nx, 2);
	dc.col(0) = arma::conv_to<cx_vec>::from(x);

	dc.print();

	std::cout << typeid(x).name() << std::endl;
	std::cout << typeid(z).name() << std::endl;
	std::cout << typeid(dc).name() << std::endl;

	auto cv = zero<arma::cx_vec4>();
	cv.print();

	auto z = zero<std::complex<double>>();
	std::cout << z << std::endl;
	return 0;
}
