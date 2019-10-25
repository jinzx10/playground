#include <complex>
#include <armadillo>
#define MKL_Complex16 std::complex<double>
#include <iostream>
#include <mkl.h>
#include <cmath>

using namespace std;
using namespace arma;

const std::complex<double> I(0.0, 1.0);

int main() {
	cx_mat sy = zeros<cx_mat>(2,2);
	sy(0,1) = -I;
	sy(1,0) = I;

	cx_mat tqeye = sy*sy;

	sy.print();
	tqeye.print();

	const int sz = 27;
	cx_mat a = randu(sz, sz) + I*randu(sz,sz);
	vec eigval = zeros(sz);
	a = a + a.t();

	complex<double>* tq = a.memptr(); 
	

	LAPACKE_zheevd( LAPACK_COL_MAJOR, 'V', 'U', sz, tq, sz, eigval.memptr() );

	//a.print();
	//cx_mat b = a*a.t();


	std::cout << typeid(a).name() << std::endl;
	std::cout << a.n_elem << std::endl;
	std::cout << a.n_rows << std::endl;
	std::cout << a.n_cols << std::endl;
	//(a*a.t()).print();


	return 0;
}
