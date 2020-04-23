#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <armadillo>
#include <type_traits>
#include "../utility/widgets.h"

using namespace arma;
using namespace std;

int main(int, char**argv) {

	vec a = randu(5,1);
	vec b = a;
	auto f = [] (double x) {return 1.0;};
	b.for_each([&](double& elem) {elem = f(elem);});

	b.print();

	

	

    return 0;
}
