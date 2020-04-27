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

	sp_mat a = sprandu(10,10, 0.2);
	sp_mat b = sprandu(10,10, 0.2);

	vec c(join_cols(a.diag(), b.diag()));

	c.print();

    return 0;
}
