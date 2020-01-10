#include <armadillo>
#include <type_traits>

using namespace arma;

int main() {
	arma::mat a = randu(3,3);
	arma::vec b = {1,2};

	arma::vec c = arma::conv_to<arma::vec>::from(b>1.5);

	c.print();

	std::cout << typeid(c).name() << std::endl;


	std::cout << 1.0 / (std::exp(10000000) + 1.0) << std::endl;
	std::cout << 1.0 / (std::exp(-10000000) + 1.0) << std::endl;
	std::cout << 1.0 / (std::exp(1 / arma::datum::eps) + 1.0) << std::endl;
	std::cout << 1.0 / (std::exp(1 / 1e-18) + 1.0) << std::endl;
	std::cout << arma::datum::eps << std::endl;
	


	return 0;
}
