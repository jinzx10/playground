#include <iostream>
#include <mkl.h>
#include <armadillo>

using namespace arma;

int main() {

	mat A, S, U, VT, A_pre, A_raw, A_fin, Ss;
	A_fin.load("A_fin.txt");
	A_pre.load("A_pre.txt");
	//A_raw.load("A_raw.txt");
	A_raw.load("Zt.dat");

	A = A_raw;

	std::cout << std::boolalpha << "is finite: " << std::endl 
		<< A_raw.is_finite() << std::endl
		<< A_pre.is_finite() << std::endl
		<< A_fin.is_finite() << std::endl
		<< std::endl;

	std::cout << "A_raw and A_pre diff = " << norm(A_raw-A_pre) << std::endl;
	std::cout << "A_pre and A_fin diff = " << norm(A_fin-A_pre) << std::endl;
	std::cout << std::endl << std::endl;

	///////////////////////////////////////////////////////////////////////
	//							C interface
	///////////////////////////////////////////////////////////////////////
	char jobz = 'A';
	int m = A.n_rows;
	int n = A.n_cols;

	U.set_size(m,m);
	VT.set_size(n,n);
	S.set_size(std::min(m,n));

	int lda = m;
	int ldu = m;
	int ldvt = n;

	int info;

	info = LAPACKE_dgesdd( LAPACK_COL_MAJOR, jobz, m, n, A.memptr(), lda, S.memptr(), U.memptr(), ldu, VT.memptr(), ldvt );

	Ss.zeros(m,n);
	Ss.diag() = S;
	std::cout << "C interface: info = " << info << std::endl;
	std::cout << "C dgesdd diff = " << norm(A-A_raw) << std::endl;
	std::cout << "A-U*S*VT = " << norm(A_raw-U*Ss*VT) << std::endl;
	std::cout << std::endl << std::endl;

	///////////////////////////////////////////////////////////////////////
	//						Fortran interface
	///////////////////////////////////////////////////////////////////////
	/* query and allocate the workspace */
	int lwork = -1;
	double wkopt;
	double* work;
	int* iwork = new int[8*n];
        
	A = A_raw;
	dgesdd(&jobz, &m, &n, A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, VT.memptr(), &ldvt, 
			&wkopt, &lwork, iwork, &info);

	lwork = (int)wkopt;
	work = new double[lwork];

	dgesdd(&jobz, &m, &n, A.memptr(), &lda, S.memptr(), U.memptr(), &ldu, VT.memptr(), &ldvt, 
			work, &lwork, iwork, &info);

	Ss.zeros(m,n);
	Ss.diag() = S;
	std::cout << "Fortran interface: info = " << info << std::endl;
	std::cout << "Fortran dgesdd diff = " << norm(A-A_raw) << std::endl;
	std::cout << "A-U*S*VT = " << norm(A_raw-U*Ss*VT) << std::endl;
	std::cout << std::endl << std::endl;

	return 0;
}
