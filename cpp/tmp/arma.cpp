#include <iostream>
#include <armadillo>
#include <chrono>
#include "../utility/widgets.h"
#include "../utility/arma_helper.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)

using namespace arma;


int main() {

	size_t sz = 1000;
	umat loc(2, sz, fill::zeros);
	vec val = ones(sz);
	loc.row(0) = regspace<urowvec>(0,sz-1);

	arma::arma_rng::set_seed_random();
	urowvec P = loc.row(0);
	P = shuffle(P);
	loc.row(1) = P;


	sp_mat a(loc, val, sz, sz); 

	cout << det(conv_to<mat>::from(a)) << endl;


	return 0;
}
