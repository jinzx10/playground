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

	a.print();

	arma::inplace_trans(a);
	a.print();

	uvec i = {0,1};
	uvec j = {2};
	a(i,j).eval().save("a.txt", raw_ascii);


    return 0;
}
