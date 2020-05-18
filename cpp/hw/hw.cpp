#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"
#include "../utility/math_helper.h" 

using namespace arma;
using namespace std;


int main(int, char**argv) {
	
	auto f = [] (vec const& v) -> vec {
		vec r(3);
		r(0) = v(0)*v(0)+v(1)*v(1)-1;
		r(1) = v(2)-0.5;
		r(2) = v(0)*v(0)-v(1)-0.25;
		return r;
	};

	vec x0 = {1,1,1};
	broydenroot(f, x0, "bad");

	x0.print();


    return 0;
}
