#include <armadillo>
#include <iostream>
#include "../utility/widgets.h"

using namespace arma;

int main(int, char** argv) {

	uword sz, sz_sub;
	double sparcity;
	readargs(argv, sz, sz_sub);

	sp_mat a = sprandu(sz, sz, sparcity);
	a += a.t();

	vec eigval, eval;
	mat eigvec, evec;

	Stopwatch sw;

	sw.run();
	eigs_sym(eval, evec, a, sz_sub);
	sw.report();

	sw.reset();

	sw.run();
	eig_sym(eigval, eigvec, mat(a));
	sw.report();

	return 0;

}
