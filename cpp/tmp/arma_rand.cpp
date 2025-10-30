#include <armadillo>
#include "../utility/widgets.h"

int main() {

	Stopwatch sw;

	sw.run();
	for (size_t i = 0; i != 10; ++i) {
		arma::arma_rng::set_seed_random();
	}
	sw.report();

	return 0;
}
