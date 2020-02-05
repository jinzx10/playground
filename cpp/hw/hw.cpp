#include <iostream>
#include <armadillo>

using namespace arma;

template <typename eT>
eT getfirst(Mat<eT>& m) {
	return m(0);
}

int main() {
	vec a = randu(3);
	a.print();
	std::cout << getfirst(a) << std::endl;
	std::cout << typeid(a).name() << std::endl;

	return 0;
}
