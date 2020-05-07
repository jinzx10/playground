#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

using namespace arma;
using namespace std;

int main(int, char**argv) {

	arma::arma_rng::set_seed_random();
	mat a = randu(3,3);

	mat b = a.t()*a;

	mat c = arma::sqrtmat_sympd(b);

	std::cout << c.diag()(abs(c.diag()).index_max()) << std::endl;

	c.print();



    return 0;
}
