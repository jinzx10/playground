#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <armadillo>
#include <type_traits>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

using namespace arma;
using namespace std;

/*
template <typename R, typename C, typename Op>
typename std::enable_if< R::is_row && C::is_col, arma::Mat<decltype(std::declval<Op>()(std::declval<R::elem_type>(), std::declval<C::elem_type>()))> >::type bcast_op(R const& r, C const& c) {
}
*/
int main(int, char**argv) {

	std::cout << "ready" << std::endl;
	std::cout << "ready2" << std::endl;
	for (int i = 0; i != 10; ++i) {
		
		printf("\033[A\33[2K\r");
		std::cout << "i = " << i << std::endl;
		std::system("sleep 0.5");
	}

	std::cout << std::endl;

    return 0;
}
