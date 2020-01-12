#define MKL_Complex16 std::complex <double>

#include <fstream>
#include <cstring>
#include <complex>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iostream>
//#include "misc/matrixop.hpp"
#include "mkl.h"

using namespace std;
//using namespace matrixop;
//#define ZGEMM zgemm_
//#define ZHEEVD zheevd_
#define ZGEMM zgemm
#define ZHEEVD zheevd


vector<std::complex<double> > zmatmat(
                                    const vector<std::complex<double> >& Mat1,
                                    const vector<std::complex<double> >& Mat2,
                                    int K, const char* opa, const char* opb){
    std::complex<double> alpha(1.0);
    std::complex<double> beta(0.0);
    int M = static_cast<int>(Mat1.size() / K);
    int N = static_cast<int>(Mat2.size() / K);
    vector<std::complex<double> > rst(M * N);
    
    ZGEMM(opa, opb, &M, &N, &K, &alpha,
          &Mat1[0], &M, &Mat2[0], &K,
          &beta, &rst[0], &M);
    
    return rst;
}

void zhediag(const vector<std::complex<double> >& Mat,
             vector<double>& eva, vector<std::complex<double> >& evt){
    int N = static_cast<int>(sqrt(Mat.size()));
    vector<std::complex<double> > evt_(Mat);
    
    int lwork(-1), lrwork(-1), liwork(-1), info(-1);
    vector<std::complex<double> > work(1);
    vector<double> rwork(1);
    vector<int> iwork(1);
    eva.resize(N);

    const char jobz = 'V';
    const char charl = 'L';
    
    ZHEEVD(&jobz, &charl, &N, &evt_[0], &N, &eva[0],
           &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
    
    lwork = (int)real(work[0]);
    lrwork = (int)rwork[0];
    liwork = iwork[0];
    work.resize(lwork);
    rwork.resize(lrwork);
    iwork.resize(liwork);
    
    ZHEEVD(&jobz, &charl, &N, &evt_[0], &N, &eva[0],
           &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
    evt = evt_;
}

int main(int argc, char** argv) {

    ifstream inp("binaryr.bin", ios::binary | ios::in);

    vector<double> v;
    const int N = 27;
    double tmp;
    int i = 0;
    while (i < N*N*2) {
        inp.read(reinterpret_cast<char*>(&tmp), static_cast<int>(sizeof(double) / sizeof(char)));
        v.push_back(tmp);
        i += 1;
    }
    //ioer::info(v.size());
    cout << v.size() << endl;

    vector<complex<double>> mat(N*N);
    for (i = 0; i < N*N; ++i) {
        mat[i].real(v[i]);
        mat[i].imag(v[i+N*N]);
    }

    vector<double> eva;
    vector<complex<double>> evt;
    //matrixop::eigh(mat, eva, evt);
    zhediag(mat, eva, evt);

    //auto ee = matrixop::matCmat(evt, evt, N);
    auto ee = zmatmat(evt, evt, N, "C", "N");
    int j;
    for (i = 0; i < N; ++i) {
        ee[i+i*N] -= 1.0;
    }
    //ioer::info(sum(pow(abs(ee),2)));

    double rst = 0.;
    for (auto x : ee) {
	    rst += pow(abs(x),2);
    }
    cout << rst << endl;

    return 0;
}
