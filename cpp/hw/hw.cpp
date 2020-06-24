#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <string>
#include <type_traits>
#include <cassert>
#include <armadillo>

using namespace std;
using namespace arma;

using iclock = std::chrono::high_resolution_clock;

int main(int, char**argv) {
	
	int max_n = 1;
	int n_atom = 40;

	mat lat_vec = randu(3,3);
	mat x_atom = randu(3, n_atom);
	vec coef = randu(n_atom);

	iclock::time_point start = iclock::now();

	int nt = 100;
	double res;
	for (int i = 0; i != nt; ++i) {
    	int sz = 2*max_n+1;
		irowvec v = regspace<irowvec>(-max_n, max_n);
		irowvec o = ones<irowvec>(sz);
		irowvec ox = ones<irowvec>(n_atom);

		mat grid = lat_vec * join_cols( kron(kron(o,o),v), kron(kron(o,v),o), kron(kron(v,o),o) );
		mat Aab = kron(x_atom, conv_to<rowvec>::from(ox)) - kron(conv_to<rowvec>::from(ox), x_atom);
		mat T = kron(grid, ones(1, size(Aab, 1))) - kron(ones(1, size(grid, 1)), Aab);
		mat Tnorm = reshape(sqrt(sum(T%T,0)), n_atom*n_atom, sz*sz*sz);
    	vec Cab = (coef*coef.t()).as_col();

    	mat CT = (erfc(Tnorm) / Tnorm).eval().each_col() % Cab;
		CT.elem( find_nonfinite(CT) ).zeros();
    	res = accu(CT);
	}
	cout << "res = " << res << endl;

	std::chrono::duration<double> dur = iclock::now() - start;
	std::cout << "time elapsed = " << dur.count() << " seconds" << std::endl;

    return 0;
}
