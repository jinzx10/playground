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

	arma::mat a = arma::randu(5,5);

	a.print();
	std::cout << std::endl;

	a(span(3,2), span(3,2)).print();
	

	return 0;
}
