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
	int sz = 10;
	int num = 100;
	
	mat a = zeros(sz,sz);
	mat eigvec;
	vec eigval;
	int count = 0;
	for (int i = 0; i != num; ++i) {
		a = randu(sz,sz);
		a = a + a.t();
		eig_sym(eigval, eigvec, a);
		a = a + (eigval(1)-eigval(0)+1) * (eigvec.col(0) * eigvec.col(0).t());
		eig_sym(eigval, eigvec, a);
		count += eigval.is_sorted();
	}

	std::cout << count << std::endl;

	return 0;
}
