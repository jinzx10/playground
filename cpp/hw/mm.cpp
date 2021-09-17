#include <iostream>
#include <armadillo>
#include "../utility/widgets.h"

using namespace std;
using namespace arma;

int main() {

	size_t sz = 1000;
	mat a = randu(sz,sz);
	mat b = randu(sz,sz);
	mat c(sz,sz, fill::zeros);

	Stopwatch sw;

	sw.run();
	for (size_t i = 0; i != sz; ++i) {
		for (size_t j = 0; j != sz; ++j) {
			for (size_t k = 0; k != sz; ++k) {
				c(i,j) += a(i,k)*b(k,j);
			}
		}
	}

	sw.report();

	c.zeros();

	sw.reset();
	sw.run();
	for (size_t i = 0; i != sz; ++i) {
		for (size_t k = 0; k != sz; ++k) {
			for (size_t j = 0; j != sz; ++j) {
				c(i,j) += a(i,k)*b(k,j);
			}
		}
	}
	sw.report();



	return 0;
}
