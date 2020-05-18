#include <iostream>
#include <armadillo>

using namespace arma;
using namespace std;

template <typename T>
struct Diag
{
	void diag() {
		T* p = static_cast<T*>(this);
		arma::eig_sym(p->eigval, p->eigvec, p->H);
	}
	vec eigval;
	mat eigvec;
};

template <typename T>
struct SumVal
{
	double sum() {
		T* p = static_cast<T*>(this);
		return arma::sum(p->eigval);
	}
};

struct Model: public Diag<Model>, public SumVal<Model>
{
	Model(mat const& h): H(h) {}
	mat H;

	vec eigval;
	mat eigvec;
};

int main() {

	mat h = {{2,1},{1,0}};

	Model m(h);

	m.diag();

	m.eigval.print();

	cout << m.sum() << endl;


	return 0;
}

