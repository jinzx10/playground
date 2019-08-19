#include <armadillo>

double isum(arma::vec2 const& v) {
	return v(0)+v(1);
}

int main() {
	arma::vec2 v2 = {1,2};

	arma::vec2 u2 = v2 + 1;

	u2.print();

	return 0;
}
