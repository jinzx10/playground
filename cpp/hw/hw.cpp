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

	uword sz, nt;

	readargs(argv, sz, nt);

	vec a = linspace<vec>(1, sz, sz);
	rowvec b = linspace<rowvec>(0, sz-1, sz);

	Stopwatch sw;

	auto f = [](vec const& v, rowvec const& r) { return bcast_plus(v,r); };

	sw.timeit(nt, f, a, b);

	mat c = std::plus<>()(repmat(a, 1, b.n_elem).eval().each_row(), b);

	a.print();
	std::cout << std::endl;
	b.print();
	std::cout << std::endl;
	c.print();
	
	pow(a, 3).print();

	

    return 0;
}
