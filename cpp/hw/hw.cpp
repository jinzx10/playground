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
	
	vec a = {1,2,3};
	rowvec b = {0, 1.1, 2.2, 3.3};

	cout << "a = " << endl;
	a.print();
	cout << endl;

	cout << "b = " << endl;
	b.print();
	cout << endl;

	bcast_op<'+'>(a,b).print();
	bcast_op<'-'>(a,b).print();
	bcast_op<'*'>(a,b).print();
	bcast_op<'/'>(a,b).print();

	bcast_op<'+'>(b,a).print();
	bcast_op<'-'>(b,a).print();
	bcast_op<'*'>(b,a).print();
	bcast_op<'/'>(b,a).print();


    return 0;
}
