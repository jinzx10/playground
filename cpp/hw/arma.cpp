#include <iostream>
#include <armadillo>
#include <chrono>

using iclock = std::chrono::high_resolution_clock;
using namespace arma;

int main() {
	uword sz = 5;
	mat a = randu(sz,sz);

	span i = span(0,2);

	return 0;
}
