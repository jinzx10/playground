#include <armadillo>
#include <iostream>

using namespace arma;

int main() {

	mat a = randu(3,3);
	a.print();
	mat b;
	b = std::move(a);
	b.print();
	a.print();
	return 0;
}
