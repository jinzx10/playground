#include <cmath>
#include <complex>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <vector>

#define MKL_Complex16 std::complex<double>
#include <mkl.h>

using namespace std;
using namespace arma;

const std::complex<double> I(0.0, 1.0);

int main() {
	const int sz = 27;

	/////////////////////////////////////////////////////////////
	//					read matrix	
	/////////////////////////////////////////////////////////////
    ifstream inp("binaryr.bin", ios::binary | ios::in);
	int i = 0;
    double tmp;
    vector<double> v;
    while (i < sz*sz*2) {
        inp.read(reinterpret_cast<char*>(&tmp), static_cast<int>(sizeof(double) / sizeof(char)));
        v.push_back(tmp);
        i += 1;
    }

	cx_mat tq = zeros<cx_mat>(sz, sz);
    for (int i = 0; i < sz*sz; ++i) {
        tq.memptr()[i].real(v[i]);
        tq.memptr()[i].imag(v[i+sz*sz]);
    }

	auto a = tq;
	auto b = tq;
	//tq.print();

	/////////////////////////////////////////////////////////////
	//					lapacke_zheevd
	/////////////////////////////////////////////////////////////
	vec eigval = zeros(sz);
	LAPACKE_zheevd( LAPACK_COL_MAJOR, 'V', 'U', sz, a.memptr(), sz, eigval.memptr() );
	cx_mat diff_a = a*a.t() - eye(sz,sz);
	std::cout << "LAPACKE_zheev: deviation from identity: " << arma::accu(diff_a % diff_a) << std::endl;
	eigval.print();

	/////////////////////////////////////////////////////////////
	//					fortran zheevd
	/////////////////////////////////////////////////////////////
	char jobz = 'V';
	char uplo = 'L';

	/* query and allocate the workspace */
    int lwork(-1), lrwork(-1), liwork(-1), info(-1);
    vector<std::complex<double> > work(1);
    vector<double> rwork(1);
    vector<int> iwork(1);

	zheevd(&jobz, &uplo, &sz, b.memptr(), &sz, eigval.memptr(), 
			&work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);

    lwork = (int)real(work[0]);
    lrwork = (int)rwork[0];
    liwork = iwork[0];
    work.resize(lwork);
    rwork.resize(lrwork);
    iwork.resize(liwork);

	/* solve the eigenvalue problem */
	zheevd(&jobz, &uplo, &sz, b.memptr(), &sz, eigval.memptr(), 
			&work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);

	cx_mat diff_b = b*b.t() - eye(sz,sz);
	std::cout << "(fortran interface) zheevd: deviation from identity: " << arma::accu(diff_b % diff_b) << std::endl;
	eigval.print();

	return 0;
}
