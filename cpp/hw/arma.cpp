#include <armadillo>

double isum(arma::vec2 const& v) {
	return v(0)+v(1);
}

int main() {
	arma::vec2 v2 = {1,2};
	arma::vec v = {1,4,5};
	arma::vec u = {1};
	arma::vec w = {4,5};

	std::cout << isum(w) << std::endl;
	std::cout << isum(u) << std::endl;
	std::cout << isum(v) << std::endl;

	return 0;
}
