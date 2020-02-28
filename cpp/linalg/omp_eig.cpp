#include <armadillo>
#include <omp.h>
#include "../utility/widgets.h"

using namespace arma;

int main(int, char**argv) {

	int sz, nt;
	readargs(argv, sz, nt);
	Stopwatch sw;

	mat a = randu(sz, sz);
	a += a.t();

	mat eigvec(sz,sz);
	vec eigval(sz);

	sw.run();
#pragma omp parallel for firstprivate(a,eigval,eigvec)
	for (int i = 0; i < nt; ++i) {
		eig_sym(eigval, eigvec, a);
	}

	sw.report();

	return 0;
}
