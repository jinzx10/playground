#include <armadillo>

using namespace std;
using namespace arma;

int main() {

    mat a = randu(3,3);
    a += a.t();

    mat eigvec, eigval(3,3,fill::zeros);
    vec tmp(eigval.colptr(0), 3, false);

    eig_sym(tmp, eigvec, a);

    a.print();
    eigval.print();


    return 0;
}
