#include <armadillo>

using namespace arma;

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

	return 0;
}
